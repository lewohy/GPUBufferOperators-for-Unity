#ifndef CS_GPU_ONE_SWEEP_PREFIX_SCAN_HLSL
#define CS_GPU_ONE_SWEEP_PREFIX_SCAN_HLSL

//#pragma kernel ClearBuffer
//#pragma kernel PrefixScan

//#pragma use_dxc
//#pragma require wavebasic
//#pragma require waveballot

#define NUM_GROUP_THREADS 128

// macro used for computing bank-conflict-free shared memory array indices
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define SHARED_MEMORY_ADDRESS(n) ((n) + CONFLICT_FREE_OFFSET(n))

#define FLAG uint
#define PARTITION_DESCRIPTOR uint
static const FLAG FLAG_INVALID   = 0;
static const FLAG FLAG_AGGREGATE = 1;
static const FLAG FLAG_PREFIX    = 2;
static const uint partition_descriptor_flag_mask = 0x00000003;
static const uint partition_descriptor_value_shift = 2;
static const uint parallelized_look_back_count = 16; // WaveGetLaneCount() must be more than or equal to this value

RWStructuredBuffer<uint> data_buffer;
// As far as I tested, (rw_structured_buffer + atomic_add) is faster than (counter_buffer + increment_counter).
globallycoherent RWStructuredBuffer<uint> partition_index_buffer;
globallycoherent RWStructuredBuffer<PARTITION_DESCRIPTOR> partition_descriptor_buffer;

uint num_elements;
uint thread_offset;
bool is_inclusive_scan;

static const uint num_group_threads = NUM_GROUP_THREADS;
static const uint num_elements_per_group = 2u * NUM_GROUP_THREADS;
static const uint num_elements_per_group_minus_1 = num_elements_per_group - 1u;
static const uint sma_num_elements_per_group = SHARED_MEMORY_ADDRESS(num_elements_per_group);
static const uint sma_num_elements_per_group_minus_1 = SHARED_MEMORY_ADDRESS(num_elements_per_group_minus_1);

static const uint s_scan_len = sma_num_elements_per_group + 1u;

groupshared uint s_scan[s_scan_len];
groupshared uint s_partition_index;
groupshared uint s_prev_reduction;

inline uint GetPartitionIndex(uint group_thread_id)
{
    if (group_thread_id == 0u)
    {
        InterlockedAdd(partition_index_buffer[0], 1u, s_partition_index);
    }
    GroupMemoryBarrierWithGroupSync();

    return s_partition_index;
}

inline PARTITION_DESCRIPTOR CreatePartitionDescriptor(uint value, FLAG flag)
{
    return value << partition_descriptor_value_shift | flag;
}

inline FLAG GetPartitionDescriptorFlag(PARTITION_DESCRIPTOR partition_descriptor)
{
    return partition_descriptor & partition_descriptor_flag_mask;
}

inline uint GetPartitionDescriptorValue(PARTITION_DESCRIPTOR partition_descriptor)
{
    return partition_descriptor >> partition_descriptor_value_shift;
}

inline uint Lookback(uint group_thread_id, uint partition_index)
{
    if (partition_index == 0u)
    {
        return 0u;
    }

    uint prev_reduction = 0u;
    uint look_back_id = partition_index;

    while (true)
    {
        if (group_thread_id < min(parallelized_look_back_count, look_back_id))
        {
            const PARTITION_DESCRIPTOR partition_descriptor = partition_descriptor_buffer[look_back_id - group_thread_id - 1u];

            const FLAG flag = GetPartitionDescriptorFlag(partition_descriptor);

            if (!WaveActiveAnyTrue(flag == FLAG_INVALID)) // if all partition descriptors are aggregate or prefix
            {
                const uint partition_descriptor_value = GetPartitionDescriptorValue(partition_descriptor);

                if (WaveActiveAllTrue(flag == FLAG_AGGREGATE)) // if all partition descriptors are aggregate
                {
                    prev_reduction += WaveActiveSum(partition_descriptor_value);
                    look_back_id -= parallelized_look_back_count;
                }
                else
                {
                    const uint lowest_prefix_wave_index = firstbitlow(WaveActiveBallot(flag == FLAG_PREFIX).x);
                    if (group_thread_id <= lowest_prefix_wave_index)
                    {
                        prev_reduction += WaveActiveSum(partition_descriptor_value);
                        if (group_thread_id == 0)
                        {
                            InterlockedAdd(partition_descriptor_buffer[partition_index], CreatePartitionDescriptor(prev_reduction, FLAG_PREFIX - FLAG_AGGREGATE));
                            s_prev_reduction = prev_reduction;
                        }
                    }
                    break;
                }
            }
        }
        else
        {
            break;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    return s_prev_reduction;
}

[numthreads(NUM_GROUP_THREADS, 1, 1)]
void ClearBuffer(uint thread_id : SV_DispatchThreadID)
{
    thread_id += thread_offset;
    if (thread_id < num_elements)
    {
        partition_descriptor_buffer[thread_id] = 0u;
    }
    if (thread_id == 0u)
    {
        partition_index_buffer[0] = 0u;
    }
}

[numthreads(NUM_GROUP_THREADS, 1, 1)]
void PrefixScan(uint group_thread_id : SV_GroupThreadID)
{
    const uint partition_index = GetPartitionIndex(group_thread_id);

    // handle two values in one thread
    uint ai = group_thread_id;
    uint bi = ai + num_group_threads;
    ai = SHARED_MEMORY_ADDRESS(ai);
    bi = SHARED_MEMORY_ADDRESS(bi);

    uint global_ai = group_thread_id + num_elements_per_group * partition_index;
    uint global_bi = global_ai + num_group_threads;

    // copy input data to shared memory
    s_scan[ai] = global_ai < num_elements ? data_buffer[global_ai] : 0u;
    s_scan[bi] = global_bi < num_elements ? data_buffer[global_bi] : 0u;

    uint offset = 1u;

    // upsweep step
    for (uint du = num_elements_per_group >> 1u; du > 0u; du >>= 1u)
    {
        GroupMemoryBarrierWithGroupSync();

        if (group_thread_id < du)
        {
            uint ai_u = offset * ((group_thread_id << 1u) + 1u) - 1u;
            uint bi_u = offset * ((group_thread_id << 1u) + 2u) - 1u;
            ai_u = SHARED_MEMORY_ADDRESS(ai_u);
            bi_u = SHARED_MEMORY_ADDRESS(bi_u);

            s_scan[bi_u] += s_scan[ai_u];
        }

        offset <<= 1u;
    }

    // save the total sum on global memory
    if (group_thread_id == 0u)
    {
        const uint group_sum = s_scan[sma_num_elements_per_group_minus_1];
        InterlockedCompareStore(partition_descriptor_buffer[partition_index], 0u, CreatePartitionDescriptor(group_sum, partition_index == 0u ? FLAG_PREFIX : FLAG_AGGREGATE));
        s_scan[sma_num_elements_per_group_minus_1] = 0u;
        s_scan[sma_num_elements_per_group] = group_sum;
    }

    const uint prev_reduction = Lookback(group_thread_id, partition_index);

    // downsweep step
    for (uint dd = 1u; dd < num_elements_per_group; dd <<= 1u)
    {
        offset >>= 1u;

        GroupMemoryBarrierWithGroupSync();

        if (group_thread_id < dd)
        {
            uint ai_d = offset * ((group_thread_id << 1u) + 1u) - 1u;
            uint bi_d = offset * ((group_thread_id << 1u) + 2u) - 1u;
            ai_d = SHARED_MEMORY_ADDRESS(ai_d);
            bi_d = SHARED_MEMORY_ADDRESS(bi_d);

            uint temp = s_scan[ai_d];
            s_scan[ai_d] = s_scan[bi_d];
            s_scan[bi_d] += temp;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (is_inclusive_scan)
    {
        ai = group_thread_id + 1u;
        bi = ai + num_group_threads;
        ai = SHARED_MEMORY_ADDRESS(ai);
        bi = SHARED_MEMORY_ADDRESS(bi);
    }

    // copy scanned data to global memory
    if (global_ai < num_elements)
        data_buffer[global_ai] = s_scan[ai] + prev_reduction;
    if (global_bi < num_elements)
        data_buffer[global_bi] = s_scan[bi] + prev_reduction;
}


#endif /* CS_GPU_ONE_SWEEP_PREFIX_SCAN_HLSL */