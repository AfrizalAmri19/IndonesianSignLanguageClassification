  *	m?????Z@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate????"2??!LS???M@)?#d ?.??1r)SjL@:Preprocessing2T
Iterator::Root::ParallelMapV2K<?l???!??Q?j+@)K<?l???1??Q?j+@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?????Պ?!?????s(@)g???ْ??1	???#@:Preprocessing2E
Iterator::Root?/Ie?9??!???'?6@)?#?]J]??1????? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??j,a??!??vB{S@)a??>??t?12?~??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?5x_?e?!???D-@)?5x_?e?1???D-@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapJ}Yک???!?!??{N@)_????`?1??Y????:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor'??bW?!lk?ht??)'??bW?1lk?ht??:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??rf?BO?!?߁?=|??)??rf?BO?1?߁?=|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.