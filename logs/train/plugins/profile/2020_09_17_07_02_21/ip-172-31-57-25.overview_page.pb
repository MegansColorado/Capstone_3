�	�:M�r@�:M�r@!�:M�r@	N̳��`�?N̳��`�?!N̳��`�?"n
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
	>x�҆�@>x�҆�@!>x�҆�@      ��!       "	�b('Z�r@�b('Z�r@!�b('Z�r@*      ��!       2      ��!       :	���-=��?���-=��?!���-=��?B      ��!       J	8��+�F�?8��+�F�?!8��+�F�?R      ��!       Z	8��+�F�?8��+�F�?!8��+�F�?JGPUYO̳��`�?b �"
Ugradient_tape/functional_15/conv2d_transpose_28/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter���f~�?!���f~�?"c
Ggradient_tape/functional_15/conv2d_transpose_28/conv2d_transpose/Conv2DConv2D🦟�۵?!�7��(l�?"[
2functional_15/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput�|�2C�?!���z~�?"k
Agradient_tape/functional_15/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterNJ�u��?!���Я<�?"
Ugradient_tape/functional_15/conv2d_transpose_27/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter?��O.�?!�9��5��?"a
@gradient_tape/functional_15/max_pooling2d_12/MaxPool/MaxPoolGradMaxPoolGradtI�3^O�?!�"a!��?"[
2functional_15/conv2d_transpose_29/conv2d_transposeConv2DBackpropInput|�a��?!$_��	�?"
Ugradient_tape/functional_15/conv2d_transpose_29/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�6����?!��g��)�?"V
8gradient_tape/functional_15/conv2d_transpose_28/ReluGradReluGradj�A	��?!y�B��?"k
Agradient_tape/functional_15/conv2d_20/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�h��?!�h�7�?Q      Y@YΎZ��5@aW�7�LW@qF�G�dG@y\�n�;]?"�	
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�46.7867% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 