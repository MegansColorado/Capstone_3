�	s�V{�J@s�V{�J@!s�V{�J@	��O�Z�?��O�Z�?!��O�Z�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6s�V{�J@}(@1`>Y1\�E@AT�J�ó�?I�/���9�?YO;�5Y��?*	�Zd;�]@2F
Iterator::Modelh��n�?!-Ab�qA@)2�F� �?1� m�6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat@4���?!=C�;H�;@)�\T��?1�^��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatem��?!V�f�rM9@)o���?1S���x3@:Preprocessing2U
Iterator::Model::ParallelMapV2����y�?!���f3�'@)����y�?1���f3�'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"��T2 �?!�y�N>GP@)&���?1��N8 @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn��4҂?!�pz��@)n��4҂?1�pz��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��>s֧|?!$�]�R@)��>s֧|?1$�]�R@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap:�V�S�?!�����P=@)��Iطs?1�DG�[@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��O�Z�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	}(@}(@!}(@      ��!       "	`>Y1\�E@`>Y1\�E@!`>Y1\�E@*      ��!       2	T�J�ó�?T�J�ó�?!T�J�ó�?:	�/���9�?�/���9�?!�/���9�?B      ��!       J	O;�5Y��?O;�5Y��?!O;�5Y��?R      ��!       Z	O;�5Y��?O;�5Y��?!O;�5Y��?JGPUY��O�Z�?b �"~
Tgradient_tape/functional_9/conv2d_transpose_16/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter���6��?!���6��?"b
Fgradient_tape/functional_9/conv2d_transpose_16/conv2d_transpose/Conv2DConv2D D_�A��?!�;8lW0�?"Z
1functional_9/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput!�(�ȭ?!˸�?Q�?"_
>gradient_tape/functional_9/max_pooling2d_6/MaxPool/MaxPoolGradMaxPoolGrad!/s��f�?!�l���?"j
@gradient_tape/functional_9/conv2d_12/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����?!a�}M�{�?"U
7gradient_tape/functional_9/conv2d_transpose_16/ReluGradReluGrad���@ڠ?!R�k ��?"Z
1functional_9/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput�C�zo��?!Ř/[���?"K
-gradient_tape/functional_9/conv2d_11/ReluGradReluGrad��a{��?!�ʛꐻ�?"~
Tgradient_tape/functional_9/conv2d_transpose_17/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterz���6�?!(�}�a��?"j
@gradient_tape/functional_9/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter���_� �?!�:E>��?Q      Y@Y<Eg@()@aY����U@q�3�G	:@y@�̭w��?"�	
both�Your program is POTENTIALLY input-bound because 13.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�26.0362% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 