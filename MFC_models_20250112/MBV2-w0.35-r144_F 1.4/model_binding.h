/* U-TOE Generated File */ 
#ifndef MODEL_BINDING_H 
#define MODEL_BINDING_H 
static inline void model_bind_tvm_iovars(const mlmodel_t *model, struct tvmgen_default_inputs *inputs, struct tvmgen_default_outputs *outputs) { 
inputs->conv2d_1 = model->params[0].values; 
inputs->weight_1x1_conv2d_1 = model->params[1].values; 
inputs->weight_depth_wise_1 = model->params[2].values; 
inputs->weight_1x1_conv2d_linear_1 = model->params[3].values; 
inputs->weight_1x1_conv2d_2 = model->params[4].values; 
inputs->weight_depth_wise_2 = model->params[5].values; 
inputs->weight_1x1_conv2d_linear_2 = model->params[6].values; 
inputs->weight_1x1_conv2d_3 = model->params[7].values; 
inputs->weight_depth_wise_3 = model->params[8].values; 
inputs->weight_1x1_conv2d_linear_3 = model->params[9].values; 
inputs->weight_1x1_conv2d_4 = model->params[10].values; 
inputs->weight_depth_wise_4 = model->params[11].values; 
inputs->weight_1x1_conv2d_linear_4 = model->params[12].values; 
inputs->weight_1x1_conv2d_5 = model->params[13].values; 
inputs->weight_depth_wise_5 = model->params[14].values; 
inputs->weight_1x1_conv2d_linear_5 = model->params[15].values; 
inputs->weight_1x1_conv2d_6 = model->params[16].values; 
inputs->weight_depth_wise_6 = model->params[17].values; 
inputs->weight_1x1_conv2d_linear_6 = model->params[18].values; 
inputs->weight_1x1_conv2d_7 = model->params[19].values; 
inputs->weight_depth_wise_7 = model->params[20].values; 
inputs->weight_1x1_conv2d_linear_7 = model->params[21].values; 
inputs->weight_1x1_conv2d_8 = model->params[22].values; 
inputs->weight_depth_wise_8 = model->params[23].values; 
inputs->weight_1x1_conv2d_linear_8 = model->params[24].values; 
inputs->weight_1x1_conv2d_9 = model->params[25].values; 
inputs->weight_depth_wise_9 = model->params[26].values; 
inputs->weight_1x1_conv2d_linear_9 = model->params[27].values; 
inputs->weight_1x1_conv2d_10 = model->params[28].values; 
inputs->weight_depth_wise_10 = model->params[29].values; 
inputs->weight_1x1_conv2d_linear_10 = model->params[30].values; 
inputs->weight_1x1_conv2d_11 = model->params[31].values; 
inputs->weight_depth_wise_11 = model->params[32].values; 
inputs->weight_1x1_conv2d_linear_11 = model->params[33].values; 
inputs->weight_1x1_conv2d_12 = model->params[34].values; 
inputs->weight_depth_wise_12 = model->params[35].values; 
inputs->weight_1x1_conv2d_linear_12 = model->params[36].values; 
inputs->weight_1x1_conv2d_13 = model->params[37].values; 
inputs->weight_depth_wise_13 = model->params[38].values; 
inputs->weight_1x1_conv2d_linear_13 = model->params[39].values; 
inputs->weight_1x1_conv2d_14 = model->params[40].values; 
inputs->weight_depth_wise_14 = model->params[41].values; 
inputs->weight_1x1_conv2d_linear_14 = model->params[42].values; 
inputs->weight_1x1_conv2d_15 = model->params[43].values; 
inputs->weight_depth_wise_15 = model->params[44].values; 
inputs->weight_1x1_conv2d_linear_15 = model->params[45].values; 
inputs->weight_1x1_conv2d_16 = model->params[46].values; 
inputs->weight_depth_wise_16 = model->params[47].values; 
inputs->weight_1x1_conv2d_linear_16 = model->params[48].values; 
inputs->weight_1x1_conv2d_17 = model->params[49].values; 
inputs->weight_depth_wise_17 = model->params[50].values; 
inputs->weight_1x1_conv2d_linear_17 = model->params[51].values; 
inputs->conv2d_2 = model->params[52].values; 
inputs->data = model->input_vars[0].values; 
outputs->output = model->output_vars[0].values; 
}
#endif