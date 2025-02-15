�	F;�ICX@F;�ICX@!F;�ICX@	Q�
X��?Q�
X��?!Q�
X��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-F;�ICX@���n��?1�K����W@I���x��?Y�`���?*	�E����[@2F
Iterator::Model�S:X��?!�ݳ��E@)ްmQf��?1�Ê���8@:Preprocessing2U
Iterator::Model::ParallelMapV2��_�L�?!�c/���2@)��_�L�?1�c/���2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����v�?!��홍�8@)�1ZGU�?15X!�"�1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�SV��D�?!��a��c4@)���Ց?1�.U"A/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Z�a/�?!-�"LI-L@)uʣaQ�?1w���$Y@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���hqƀ?!'�1K�e@)���hqƀ?1'�1K�e@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice9�	�ʼu?!mk��@)9�	�ʼu?1mk��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���~1�?!=��ǻ�7@){g�UIdo?1���y�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Q�
X��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���n��?���n��?!���n��?      ��!       "	�K����W@�K����W@!�K����W@*      ��!       2      ��!       :	���x��?���x��?!���x��?B      ��!       J	�`���?�`���?!�`���?R      ��!       Z	�`���?�`���?!�`���?JGPUYQ�
X��?b �"~
Tgradient_tape/functional_7/conv2d_transpose_14/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterp�jܗI�?!p�jܗI�?"b
Fgradient_tape/functional_7/conv2d_transpose_14/conv2d_transpose/Conv2DConv2D���+���?!x3&����?"Z
1functional_7/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput����K�?!4[ D���?"j
@gradient_tape/functional_7/conv2d_13/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterviu����?!�ھ�ap�?"9
functional_7/dense_7/BiasAddBiasAdd��G
_��?!�Xc�7,�?"i
Lfunctional_7/dense_7/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown��#��?!��8��?"�
xgradient_tape/functional_7/conv2d_transpose_14/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown������?!p���W�?"V
:gradient_tape/functional_7/dense_7/Tensordot/MatMul/MatMulMatMulQq�勜?!�j]����?"=
functional_7/conv2d_13/Relu_FusedConv2D�P�t�9�?!�}���?"h
?gradient_tape/functional_7/conv2d_13/Conv2D/Conv2DBackpropInputConv2DBackpropInput|9xW��?!O?6Օ��?Q      Y@Y����=1@a��=��T@q��T3�mA@y��~���m?"�	
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�34.8574% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 