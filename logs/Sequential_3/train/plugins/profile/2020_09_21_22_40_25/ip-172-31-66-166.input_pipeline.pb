	V�a@�k@V�a@�k@!V�a@�k@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-V�a@�k@��aۢ�?1�PlM$k@A1�䠄y?Ic'��� @*	/�$�`@2F
Iterator::Model���Z`��?!���ݵB@)��@��_�?1M����7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ���7�?!7j8���:@)����W;�?1�'��33@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatef�"��)�?!$9���7@)` ��c�?1���1@:Preprocessing2U
Iterator::Model::ParallelMapV2\:�<c_�?!�,Hl��*@)\:�<c_�?1�,Hl��*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��E_�?!GNh"JO@)�����?1�핓.� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorM֨�h�?!MqE���@)M֨�h�?1MqE���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��yT�߁?!
�|g+@)��yT�߁?1
�|g+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�O �Ȣ?!u;)�4�;@)�]i��t?1��4q�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��aۢ�?��aۢ�?!��aۢ�?      ��!       "	�PlM$k@�PlM$k@!�PlM$k@*      ��!       2	1�䠄y?1�䠄y?!1�䠄y?:	c'��� @c'��� @!c'��� @B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 