�	��'�W@��'�W@!��'�W@	S�7��X�?S�7��X�?!S�7��X�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��'�W@����̓�?1��t��8W@I�tۈg�?Y����?*	��/�$^@2F
Iterator::Modeld�]K��?!�EgA@)ٗl<�b�?1 ���2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����>�?!Gˆ8(^;@)� ��q4�?1*���@�2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�1^�?!�YF��8@)r�#D�?1!܍}�2@:Preprocessing2U
Iterator::Model::ParallelMapV2� OZ���?!C���?.@)� OZ���?1C���?.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���r-Z�?!�w|]�{P@)�9y�	��?1X�IH��!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�?�,�?!:�2�%!@)�?�,�?1:�2�%!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicev7Ou�̀?!7�="'8@)v7Ou�̀?17�="'8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapp_�Q�?!ZF���=@)��v� �w?1��L�S@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9S�7��X�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����̓�?����̓�?!����̓�?      ��!       "	��t��8W@��t��8W@!��t��8W@*      ��!       2      ��!       :	�tۈg�?�tۈg�?!�tۈg�?B      ��!       J	����?����?!����?R      ��!       Z	����?����?!����?JGPUYS�7��X�?b �"}
Sgradient_tape/functional_3/conv2d_transpose_2/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter:%nkd��?!:%nkd��?"a
Egradient_tape/functional_3/conv2d_transpose_2/conv2d_transpose/Conv2DConv2D����Q��?!f�'�X�?"Y
0functional_3/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput��S�?!dt�l�-�?"i
?gradient_tape/functional_3/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterGq�uȤ�?![��v+�?"9
functional_3/dense_1/BiasAddBiasAdd�n�9)�?!�ɴc
��?"i
Lfunctional_3/dense_1/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown��ɰ�[�?!Rf��ȃ�?"�
wgradient_tape/functional_3/conv2d_transpose_2/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown��T�d3�?!������?"V
:gradient_tape/functional_3/dense_1/Tensordot/MatMul/MatMulMatMulۜ�O�@�?!|V{��?"<
functional_3/conv2d_3/Relu_FusedConv2D�����?!����?"A
%functional_3/dense_1/Tensordot/MatMulMatMul?��
��?!v��t�?Q      Y@Y����=1@a��=��T@qb��Ǚ�@y���c�2n?"�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 