	����1k@����1k@!����1k@	I��-�?I��-�?!I��-�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����1k@M��E��?1��Z��j@I)��=$��?Y<O<g�?*	o��ʙ_@2F
Iterator::Model�/Ie�9�?!F��k-�B@)D��]L3�?1�ؓ]A�6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(z�c��?!�1�y��<@)��TN{�?1	|�u4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF~�,�?!�z��5@)/�H��?1n�kN@0@:Preprocessing2U
Iterator::Model::ParallelMapV2���l�?�?!����2�-@)���l�?�?1����2�-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-y<-?�?!�cM��HO@)z ���!�?1�ۈCj!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�P���ʅ?!�PL-� @)�P���ʅ?1�PL-� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice[�� ��|?!�Py�r@)[�� ��|?1�Py�r@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapLqU�wE�?! (4�M$9@)G���R{q?1��Ѝ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9I��-�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	M��E��?M��E��?!M��E��?      ��!       "	��Z��j@��Z��j@!��Z��j@*      ��!       2      ��!       :	)��=$��?)��=$��?!)��=$��?B      ��!       J	<O<g�?<O<g�?!<O<g�?R      ��!       Z	<O<g�?<O<g�?!<O<g�?JGPUYI��-�?b 