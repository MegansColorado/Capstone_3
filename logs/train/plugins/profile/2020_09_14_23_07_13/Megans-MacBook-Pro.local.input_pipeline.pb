	�l����@�l����@!�l����@	��>W�c?��>W�c?!��>W�c?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�l����@���(\��?Aj�t3�@Y��/�$�?*	     ��@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�$��C�?!|��@P@)w��/��?1h|� ZH@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�$��C�?!�?㋍ 7@)�A`��"�?1QԀ9��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate�Q����?!bR��<.@)�n����?1�{�v�.@:Preprocessing2F
Iterator::Model����K�?!YB����#@);�O��n�?1�����@:Preprocessing2U
Iterator::Model::ParallelMapV2��~j�t�?!a�o�(j @)��~j�t�?1a�o�(j @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��~j�t�?!a�o�(j @)����Mb�?1DkbR��?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�I+��?!¾����?)�I+��?1¾����?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!s�����?)�~j�t�h?1s�����?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range����MbP?!DkbR��?)����MbP?1DkbR��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[1]::FromTensor����MbP?!DkbR��?)����MbP?1DkbR��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate[0]::TensorSlice����MbP?!DkbR��?)����MbP?1DkbR��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��>W�c?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���(\��?���(\��?!���(\��?      ��!       "      ��!       *      ��!       2	j�t3�@j�t3�@!j�t3�@:      ��!       B      ��!       J	��/�$�?��/�$�?!��/�$�?R      ��!       Z	��/�$�?��/�$�?!��/�$�?JCPU_ONLYY��>W�c?b 