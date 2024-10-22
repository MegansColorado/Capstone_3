�	4`���@4`���@!4`���@	�r��0�?�r��0�?!�r��0�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-4`���@}<�ݭ�@1��,'A֚@I�Hm�$�?YG=D�;��?*	� �rhaf@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��_���?!>�6��4@@)צ����?1>�;
9@:Preprocessing2F
Iterator::Model:d�w�?!��2�9�A@)[#�qp�?1!Sa_�r2@:Preprocessing2U
Iterator::Model::ParallelMapV2�E�~�?!jzٟ�1@)�E�~�?1jzٟ�1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�z0)>>�?!Ë�U�2@)�U-�(�?1�Qa���$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipx�'-\V�?!���1c P@)����c>�?1�V=<�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�@gҦ�?!�ť��� @)�@gҦ�?1�ť��� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����?!���u�|@)����?1���u�|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���jפ?!�8��5�6@)�'�bd�|?1jfU��f@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�r��0�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	}<�ݭ�@}<�ݭ�@!}<�ݭ�@      ��!       "	��,'A֚@��,'A֚@!��,'A֚@*      ��!       2      ��!       :	�Hm�$�?�Hm�$�?!�Hm�$�?B      ��!       J	G=D�;��?G=D�;��?!G=D�;��?R      ��!       Z	G=D�;��?G=D�;��?!G=D�;��?JGPUY�r��0�?b �"}
Sgradient_tape/functional_3/conv2d_transpose_2/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter)�5�?!)�5�?"a
Egradient_tape/functional_3/conv2d_transpose_2/conv2d_transpose/Conv2DConv2D�@F�z��?!ب7H�?"Y
0functional_3/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput��"�=��?!�"I���?"V
:gradient_tape/functional_3/dense_1/Tensordot/MatMul/MatMulMatMulP�O�?!lYhd+�?"i
?gradient_tape/functional_3/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�h"�?!�zڎ�?"A
%functional_3/dense_1/Tensordot/MatMulMatMul+f&���?!`�G���?"X
<gradient_tape/functional_3/dense_1/Tensordot/MatMul/MatMul_1MatMul�����?!�+{����?"9
functional_3/dense_1/BiasAddBiasAdd5�7�ѣ?!�n�� �?"<
functional_3/conv2d_3/Relu_FusedConv2D!"L�[2�?!1ڀ,�?"�
wgradient_tape/functional_3/conv2d_transpose_2/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown7~���b�?!U��C�?Q      Y@Y�_��_�@aZ�Z�W@q�*��?y�AS 5?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 