	��6 �X@��6 �X@!��6 �X@	���[���?���[���?!���[���?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��6 �X@���x�?1c`�X@I <���?Y��bE�?*	1�Z�]@2F
Iterator::Model¡�xxϩ?!�_RW	E@)��
���?1��y��9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8�Q��m�?!�C$i�:@)�+ٱ��?1xoҺ�-3@:Preprocessing2U
Iterator::Model::ParallelMapV2�Yh�4�?!dM+!V0@)�Yh�4�?1dM+!V0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��b.�?!v"�s�4@)p�71$'�?1}I��~8/@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�9��*��?!zP�Af@)�9��*��?1zP�Af@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�;�ı?!�d����L@)�x?n�|�?1�ܝ�"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicez���x?!���84�@)z���x?1���84�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap4-�2��?!�K��8�7@)����Un?1�JY�b�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9���[���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���x�?���x�?!���x�?      ��!       "	c`�X@c`�X@!c`�X@*      ��!       2      ��!       :	 <���? <���?! <���?B      ��!       J	��bE�?��bE�?!��bE�?R      ��!       Z	��bE�?��bE�?!��bE�?JGPUY���[���?b 