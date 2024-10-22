�	�1�@�1�@!�1�@	�g5%��?�g5%��?!�g5%��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�1�@b��4���?1a�ri|ʇ@I4�9A��?Y\�3�?O�?*	X9��~a@2U
Iterator::Model::ParallelMapV2����˚�?!FBl��+7@)����˚�?1FBl��+7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�a���L�?!�-?T<@)��u�+.�?1!���5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�\��X3�?!��t�e9@)�F�g�u�?1����)3@:Preprocessing2F
Iterator::ModelZ���а�?!�ho�?:A@)��J
,�?1��8K�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)��=$|�?!�KH1�bP@)�(�ޅ?1�o���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���'ׄ?! �\�	@)���'ׄ?1 �\�	@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�����?!��1��@)�����?1��1��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ht�3�?!2�Í+�=@)�]���x?1�,*c��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�g5%��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	b��4���?b��4���?!b��4���?      ��!       "	a�ri|ʇ@a�ri|ʇ@!a�ri|ʇ@*      ��!       2      ��!       :	4�9A��?4�9A��?!4�9A��?B      ��!       J	\�3�?O�?\�3�?O�?!\�3�?O�?R      ��!       Z	\�3�?O�?\�3�?O�?!\�3�?O�?JGPUY�g5%��?b �"{
Qgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterԱ�}��?!Ա�}��?"_
Cgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DConv2D��t�?!a)�B���?"W
.functional_1/conv2d_transpose/conv2d_transposeConv2DBackpropInputr

M�?!>�\����?"i
?gradient_tape/functional_1/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��g��ö?!�R{8���?"T
8gradient_tape/functional_1/dense/Tensordot/MatMul/MatMulMatMul�zq��?!rýg��?"V
:gradient_tape/functional_1/dense/Tensordot/MatMul/MatMul_1MatMulPS�Ǩ�?!�h��w��?"7
functional_1/dense/BiasAddBiasAdd�f��L�?!�S9��?"<
functional_1/conv2d_1/Relu_FusedConv2Df�~���?!vu�{�&�?"�
ugradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown�R�����?!pX�#�?"g
Jfunctional_1/dense/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown���+�u�?!95d�`�?Q      Y@Y�5��P*@a_Cy�U@qM��v�9@y����`H?"�	
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�25.723% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 