	�:M�r@�:M�r@!�:M�r@	N̳��`�?N̳��`�?!N̳��`�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�:M�r@>x�҆�@1�b('Z�r@I���-=��?Y8��+�F�?*	V-��`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��T�t<�?!�}̋�]@@)��ډ���?1۽{*ub8@:Preprocessing2F
Iterator::ModelG�tF^�?!=��x�v@@)F]k�SU�?1��1@:Preprocessing2U
Iterator::Model::ParallelMapV2H�}8g�?!�b���.@)H�}8g�?1�b���.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateTrN�}�?!�p�f�4@)Uܸ��ܐ?1��Q��(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipGɫsȶ?!��C��P@))�7Ӆ�?1�AR�"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�++MJA�?!P��X�!@)�++MJA�?1P��X�!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~��ŉ��?!�z:ړ� @)~��ŉ��?1�z:ړ� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/�4'/2�?!d��\P9@)+�m��w?1]V\�_@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9O̳��`�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>x�҆�@>x�҆�@!>x�҆�@      ��!       "	�b('Z�r@�b('Z�r@!�b('Z�r@*      ��!       2      ��!       :	���-=��?���-=��?!���-=��?B      ��!       J	8��+�F�?8��+�F�?!8��+�F�?R      ��!       Z	8��+�F�?8��+�F�?!8��+�F�?JGPUYO̳��`�?b 