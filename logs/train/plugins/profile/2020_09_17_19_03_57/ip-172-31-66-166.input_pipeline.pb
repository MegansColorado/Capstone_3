	���Ok@���Ok@!���Ok@	K�����?K�����?!K�����?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���Ok@`w���s�?10.s�j@Ic'���?Y�6�xͫ�?*		�Zd+]@2F
Iterator::Modelz�rK��?!zBQ��{E@)��j�?1��_��'=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���]�?!��	"�8@)L������?1����/1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�D��b�?!�E?5@)m���U�?1;2��.0@:Preprocessing2U
Iterator::Model::ParallelMapV2j���<��?! ���Ϡ+@)j���<��?1 ���Ϡ+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/�H��?!���1�L@)��]P߂?1l��Y�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorD�l����?!,�!挑@)D�l����?1,�!挑@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�h���2x?!N�EA@)�h���2x?1N�EA@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���V�?!a.�<�8@)��Z��o?1���&y
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9L�����?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	`w���s�?`w���s�?!`w���s�?      ��!       "	0.s�j@0.s�j@!0.s�j@*      ��!       2      ��!       :	c'���?c'���?!c'���?B      ��!       J	�6�xͫ�?�6�xͫ�?!�6�xͫ�?R      ��!       Z	�6�xͫ�?�6�xͫ�?!�6�xͫ�?JGPUYL�����?b 