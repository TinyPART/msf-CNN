/* U-TOE Generated File */ 
#ifndef MODEL_BINDING_H 
#define MODEL_BINDING_H 
static inline void model_bind_tvm_iovars(const mlmodel_t *model, struct tvmgen_default_inputs *inputs, struct tvmgen_default_outputs *outputs) { 
inputs->conv2d_1 = model->params[0].values; 
inputs->weight_depth_wise_1 = model->params[1].values; 
inputs->conv2d_2 = model->params[2].values; 
inputs->weight_1x1_conv2d_18 = model->params[3].values; 
inputs->weight_depth_wise_18 = model->params[4].values; 
inputs->weight_1x1_conv2d_linear_18 = model->params[5].values; 
inputs->weight_1x1_conv2d_19 = model->params[6].values; 
inputs->weight_depth_wise_19 = model->params[7].values; 
inputs->weight_1x1_conv2d_linear_19 = model->params[8].values; 
inputs->weight_1x1_conv2d_20 = model->params[9].values; 
inputs->weight_depth_wise_20 = model->params[10].values; 
inputs->weight_1x1_conv2d_linear_20 = model->params[11].values; 
inputs->weight_1x1_conv2d_21 = model->params[12].values; 
inputs->weight_depth_wise_21 = model->params[13].values; 
inputs->weight_1x1_conv2d_linear_21 = model->params[14].values; 
inputs->weight_1x1_conv2d_22 = model->params[15].values; 
inputs->weight_depth_wise_22 = model->params[16].values; 
inputs->weight_1x1_conv2d_linear_22 = model->params[17].values; 
inputs->weight_1x1_conv2d_23 = model->params[18].values; 
inputs->weight_depth_wise_23 = model->params[19].values; 
inputs->weight_1x1_conv2d_linear_23 = model->params[20].values; 
inputs->weight_1x1_conv2d_24 = model->params[21].values; 
inputs->weight_depth_wise_24 = model->params[22].values; 
inputs->weight_1x1_conv2d_linear_24 = model->params[23].values; 
inputs->weight_1x1_conv2d_25 = model->params[24].values; 
inputs->weight_depth_wise_25 = model->params[25].values; 
inputs->weight_1x1_conv2d_linear_25 = model->params[26].values; 
inputs->weight_1x1_conv2d_26 = model->params[27].values; 
inputs->weight_depth_wise_26 = model->params[28].values; 
inputs->weight_1x1_conv2d_linear_26 = model->params[29].values; 
inputs->weight_1x1_conv2d_27 = model->params[30].values; 
inputs->weight_depth_wise_27 = model->params[31].values; 
inputs->weight_1x1_conv2d_linear_27 = model->params[32].values; 
inputs->weight_1x1_conv2d_28 = model->params[33].values; 
inputs->weight_depth_wise_28 = model->params[34].values; 
inputs->weight_1x1_conv2d_linear_28 = model->params[35].values; 
inputs->weight_1x1_conv2d_29 = model->params[36].values; 
inputs->weight_depth_wise_29 = model->params[37].values; 
inputs->weight_1x1_conv2d_linear_29 = model->params[38].values; 
inputs->weight_1x1_conv2d_30 = model->params[39].values; 
inputs->weight_depth_wise_30 = model->params[40].values; 
inputs->weight_1x1_conv2d_linear_30 = model->params[41].values; 
inputs->weight_1x1_conv2d_31 = model->params[42].values; 
inputs->weight_depth_wise_31 = model->params[43].values; 
inputs->weight_1x1_conv2d_linear_31 = model->params[44].values; 
inputs->data = model->input_vars[0].values; 
outputs->output = model->output_vars[0].values; 
}
#endif