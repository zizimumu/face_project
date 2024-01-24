import tensorflow as tf

convert=tf.lite.TFLiteConverter.from_frozen_graph("mobilenet_v2_1.0_224_frozen.pb",input_arrays=["input"],output_arrays=["MobilenetV2/Predictions/Reshape_1"],
                                                  input_shapes={"input":[1,224,224,3]})
                                                  
                                              
convert.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]  #Weight quantization


#Full integer quantization of weights and activations,input and output are left in floating point.
#converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model=convert.convert()
open("mobilenet_v2_1.0_224_self_quant.tflite","wb").write(tflite_model)
print("finish!")