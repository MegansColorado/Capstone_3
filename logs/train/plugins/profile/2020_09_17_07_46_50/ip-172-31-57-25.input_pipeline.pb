	8�Gn��u@8�Gn��u@!8�Gn��u@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-8�Gn��u@����@1�~k�u@A�(B�v�u?I\Ɏ�@�@*	�Zd;7d@2F
Iterator::ModeljhwH1�?!}7��'�C@)kg{�?1g� X�7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����R�?!GW?�9@)�~��@��?1wcz��4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��×�"�?!�&��Q8@)H�]�۝?15��2@:Preprocessing2U
Iterator::Model::ParallelMapV2�Z��8��?!%y&e/@)�Z��8��?1%y&e/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$����5�?!��h
�qN@)}�;l"3�?1�ы@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����o҄?!��o%@)����o҄?1��o%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceC���|͂?!B�r!�@)C���|͂?1B�r!�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$0��{�?!�f&��=@)q9^��Iy?1��za�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����@����@!����@      ��!       "	�~k�u@�~k�u@!�~k�u@*      ��!       2	�(B�v�u?�(B�v�u?!�(B�v�u?:	\Ɏ�@�@\Ɏ�@�@!\Ɏ�@�@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 