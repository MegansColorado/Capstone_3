	�ZD��@�ZD��@!�ZD��@	����}fg?����}fg?!����}fg?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�ZD��@ˡE����?A������@Y�MbX9�?*	     �@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�z�G��?!7T�mW@)V-��?1d�KaxV@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch���Q��?!yh=��@)���Q��?1yh=��@:Preprocessing2F
Iterator::Model���S㥫?!�w�Zn@){�G�z�?1�E~�_� @:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�l����?!t��V���?);�O��n�?1*J�#��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;�O��n�?!*J�#��?);�O��n�?1*J�#��?:Preprocessing2U
Iterator::Model::ParallelMapV2y�&1��?!�ǰ��B�?)y�&1��?1�ǰ��B�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenatey�&1��?!�ǰ��B�?)9��v���?1X'$(c��?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-��?!1w�d�?)�I+��?17�
]�F�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range����MbP?!
	�Xf��?)����MbP?1
	�Xf��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[1]::FromTensor����MbP?!
	�Xf��?)����MbP?1
	�Xf��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate[0]::TensorSlice����MbP?!
	�Xf��?)����MbP?1
	�Xf��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9����}fg?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ˡE����?ˡE����?!ˡE����?      ��!       "      ��!       *      ��!       2	������@������@!������@:      ��!       B      ��!       J	�MbX9�?�MbX9�?!�MbX9�?R      ��!       Z	�MbX9�?�MbX9�?!�MbX9�?JCPU_ONLYY����}fg?b 