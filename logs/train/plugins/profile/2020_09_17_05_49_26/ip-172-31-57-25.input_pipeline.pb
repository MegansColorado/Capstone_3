	A�G�*t@A�G�*t@!A�G�*t@	�����?�����?!�����?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-A�G�*t@nj���;@1�O����s@I�����6�?Y.��e�O�?*	�&1� d@2F
Iterator::Model:�V�S�?!���'�E@)dX��G�?1����@;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_|�/��?!�Q?1 �7@)���镲�?1�L_b�g1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���6ʢ?!#�����6@)�({K9_�?1&�ޱ51@:Preprocessing2U
Iterator::Model::ParallelMapV2!t�%z�?!U���0@)!t�%z�?1U���0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensori����+�?!��;��@)i����+�?1��;��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipPU��X6�?!u)�M�'L@)����?1���G@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��g�ej�?!���cV@)��g�ej�?1���cV@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!>��x�4:@)\>���v?1���R@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�����?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	nj���;@nj���;@!nj���;@      ��!       "	�O����s@�O����s@!�O����s@*      ��!       2      ��!       :	�����6�?�����6�?!�����6�?B      ��!       J	.��e�O�?.��e�O�?!.��e�O�?R      ��!       Z	.��e�O�?.��e�O�?!.��e�O�?JGPUY�����?b 