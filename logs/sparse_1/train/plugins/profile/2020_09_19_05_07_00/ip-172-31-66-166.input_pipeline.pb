	M���$�Y@M���$�Y@!M���$�Y@	�R�8̹�?�R�8̹�?!�R�8̹�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-M���$�Y@���q���?1%Y���&Y@I�26t�?�?YT�n.�6�?*	��Q�~`@2F
Iterator::ModelvŌ�� �?!o�Tu�D@)K\Ǹ��?1�{7A�8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�����ڠ?!8�v�k�8@)��:�p�?1P>?��N4@:Preprocessing2U
Iterator::Model::ParallelMapV2VҊo(|�?!�q�ϣ0@)VҊo(|�?1�q�ϣ0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����Q�?!�N#3o-7@)A�;��?1M+���.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�:���R�?!t�ᢏ@)�:���R�?1t�ᢏ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp�����?!�~���.M@)�Z�Qf�?1����3�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��Lny?!���Hg�@)��Lny?1���Hg�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`9B��?!1����<@)I���p�p?1��@�:�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�R�8̹�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���q���?���q���?!���q���?      ��!       "	%Y���&Y@%Y���&Y@!%Y���&Y@*      ��!       2      ��!       :	�26t�?�?�26t�?�?!�26t�?�?B      ��!       J	T�n.�6�?T�n.�6�?!T�n.�6�?R      ��!       Z	T�n.�6�?T�n.�6�?!T�n.�6�?JGPUY�R�8̹�?b 