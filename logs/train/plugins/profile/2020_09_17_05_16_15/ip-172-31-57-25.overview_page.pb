�	�i�� �@�i�� �@!�i�� �@	`�ޓ��?`�ޓ��?!`�ޓ��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�i�� �@�.��?1�Ɍ�ɇ@IJ�����?Y�GnM���?*	}?5^��^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatW������?!g�4��n@@)c'���?1�4�yN8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate%=�NΠ?!C �:�:@)��~31]�?1���N�Q3@:Preprocessing2U
Iterator::Model::ParallelMapV2���v�?!�
��B�.@)���v�?1�
��B�.@:Preprocessing2F
Iterator::Model�	g��ɠ?!���:@)���B:�?1�#�;b&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Y�X"�?!9�;XR@)��Aȇ?1ԧ���"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%�j�?!�뛰"!@)%�j�?1�뛰"!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicez�3M�~�?!2�6U@)z�3M�~�?12�6U@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapӇ.�o��?!ԉ�b?@)mU�Yv?1�C�韸@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9`�ޓ��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�.��?�.��?!�.��?      ��!       "	�Ɍ�ɇ@�Ɍ�ɇ@!�Ɍ�ɇ@*      ��!       2      ��!       :	J�����?J�����?!J�����?B      ��!       J	�GnM���?�GnM���?!�GnM���?R      ��!       Z	�GnM���?�GnM���?!�GnM���?JGPUY`�ޓ��?b �"{
Qgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter��w�9;�?!��w�9;�?"_
Cgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DConv2D}��t]�?!%�K�?"W
.functional_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput�Z�"`��?!�{��c^�?"i
?gradient_tape/functional_1/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltery���?!�m���?"T
8gradient_tape/functional_1/dense/Tensordot/MatMul/MatMulMatMul$�X��?!� x'�M�?"V
:gradient_tape/functional_1/dense/Tensordot/MatMul/MatMul_1MatMul�)�B�?!������?"7
functional_1/dense/BiasAddBiasAdd/�gD�B�?!�-�;��?"<
functional_1/conv2d_1/Relu_FusedConv2DN��\��?!K{rP%�?"�
ugradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknownSG6��m�?!�-䒾 �?"g
Jfunctional_1/dense/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown����y��?!:�K`��?Q      Y@Y�5��P*@a_Cy�U@q��&uI6@y)��l�L?"�	
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�22.2869% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 