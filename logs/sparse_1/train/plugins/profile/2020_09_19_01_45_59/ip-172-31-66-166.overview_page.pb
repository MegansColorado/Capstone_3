�	��CíY@��CíY@!��CíY@	{Gz��?{Gz��?!{Gz��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��CíY@	�%qV��?1D��~�;Y@I��%!��?Y������?*	�rh��Y@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatup�x��?!�|��J@@)+����:�?1���Ϥ7@:Preprocessing2U
Iterator::Model::ParallelMapV2�W;�sԑ?!f���
f1@)�W;�sԑ?1f���
f1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��i�:�?!�:���7@)��9��ː?1�ť��c0@:Preprocessing2F
Iterator::Model��l���?!����u<@)��b('څ?1_:�1�R%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��y�Cn�?!�]P�"�Q@)�sѐ�(�?1�h���$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor}�|�.P�?!���!@)}�|�.P�?1���!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�e��E}?!���ߏ@)�e��E}?1���ߏ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+�z���?!��'[a	=@)o��\��v?1i9��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9{Gz��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
		�%qV��?	�%qV��?!	�%qV��?      ��!       "	D��~�;Y@D��~�;Y@!D��~�;Y@*      ��!       2      ��!       :	��%!��?��%!��?!��%!��?B      ��!       J	������?������?!������?R      ��!       Z	������?������?!������?JGPUY{Gz��?b �"}
Sgradient_tape/functional_3/conv2d_transpose_4/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�㛱Z:�?!�㛱Z:�?"a
Egradient_tape/functional_3/conv2d_transpose_4/conv2d_transpose/Conv2DConv2D
&���^�?!�+(���?"Y
0functional_3/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput-������?!l�I��?"i
?gradient_tape/functional_3/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterˈb�`!�?!"�5�� �?"9
functional_3/dense_2/BiasAddBiasAddS��Iy�?!���^K��?"i
Lfunctional_3/dense_2/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown�ș5��?!$*�x�?"�
wgradient_tape/functional_3/conv2d_transpose_4/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown�+�J'�?!�f�M�Y�?"<
functional_3/conv2d_6/Relu_FusedConv2D4x೉K�?!�j��I<�?"V
:gradient_tape/functional_3/dense_2/Tensordot/MatMul/MatMulMatMulU�@�s�?!�r�b��?"g
>gradient_tape/functional_3/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput���h��?!J����?Q      Y@Y�B���0@aNozӛ�T@qij�f�+@y�ƚmm?"�	
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�13.8791% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 