	���w��@���w��@!���w��@	[mi��{�?[mi��{�?![mi��{�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���w��@�/�$�?A?5^�I��@Y�E�����?*	     @�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapR���Q�?!     �I@)�C�l���?1�����LD@:Preprocessing2F
Iterator::Model�x�&1�?!fffff�5@)�&1��?1ffffff4@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map����Mb�?!������9@)ffffff�?1     �1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::ConcatenateZd;�O��?!ffffff"@)����K�?1333333"@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat#��~j��?!333333 @)��~j�t�?1ffffff@:Preprocessing2U
Iterator::Model::ParallelMapV2���Q��?!      �?)���Q��?1      �?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Q��?!      �?)���Q��?1      �?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�~j�t��?!333333�?)�~j�t��?1333333�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range{�G�zt?!      �?){�G�zt?1      �?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[1]::FromTensor����MbP?!�������?)����MbP?1�������?:Preprocessing2b
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate[0]::TensorSlice:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Zmi��{�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�/�$�?�/�$�?!�/�$�?      ��!       "      ��!       *      ��!       2	?5^�I��@?5^�I��@!?5^�I��@:      ��!       B      ��!       J	�E�����?�E�����?!�E�����?R      ��!       Z	�E�����?�E�����?!�E�����?JCPU_ONLYYZmi��{�?b 