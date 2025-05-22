/* U-TOE Generated File */ 
#ifndef MODEL_BINDING_H 
#define MODEL_BINDING_H 
static inline void model_bind_tvm_iovars(const mlmodel_t *model, struct tvmgen_default_inputs *inputs, struct tvmgen_default_outputs *outputs) { 
inputs->conv2d_1 = model->params[0].values; 
inputs->weight_depth_wise_1 = model->params[1].values; 
inputs->conv2d_2 = model->params[2].values; 
inputs->weight_1x1_conv2d_32 = model->params[3].values; 
inputs->weight_depth_wise_32 = model->params[4].values; 
inputs->weight_1x1_conv2d_linear_32 = model->params[5].values; 
inputs->weight_1x1_conv2d_33 = model->params[6].values; 
inputs->weight_depth_wise_33 = model->params[7].values; 
inputs->weight_1x1_conv2d_linear_33 = model->params[8].values; 
inputs->weight_1x1_conv2d_34 = model->params[9].values; 
inputs->weight_depth_wise_34 = model->params[10].values; 
inputs->weight_1x1_conv2d_linear_34 = model->params[11].values; 
inputs->weight_1x1_conv2d_35 = model->params[12].values; 
inputs->weight_depth_wise_35 = model->params[13].values; 
inputs->weight_1x1_conv2d_linear_35 = model->params[14].values; 
inputs->weight_1x1_conv2d_36 = model->params[15].values; 
inputs->weight_depth_wise_36 = model->params[16].values; 
inputs->weight_1x1_conv2d_linear_36 = model->params[17].values; 
inputs->weight_1x1_conv2d_37 = model->params[18].values; 
inputs->weight_depth_wise_37 = model->params[19].values; 
inputs->weight_1x1_conv2d_linear_37 = model->params[20].values; 
inputs->weight_1x1_conv2d_38 = model->params[21].values; 
inputs->weight_depth_wise_38 = model->params[22].values; 
inputs->weight_1x1_conv2d_linear_38 = model->params[23].values; 
inputs->weight_1x1_conv2d_39 = model->params[24].values; 
inputs->weight_depth_wise_39 = model->params[25].values; 
inputs->weight_1x1_conv2d_linear_39 = model->params[26].values; 
inputs->weight_1x1_conv2d_40 = model->params[27].values; 
inputs->weight_depth_wise_40 = model->params[28].values; 
inputs->weight_1x1_conv2d_linear_40 = model->params[29].values; 
inputs->weight_1x1_conv2d_41 = model->params[30].values; 
inputs->weight_depth_wise_41 = model->params[31].values; 
inputs->weight_1x1_conv2d_linear_41 = model->params[32].values; 
inputs->weight_1x1_conv2d_42 = model->params[33].values; 
inputs->weight_depth_wise_42 = model->params[34].values; 
inputs->weight_1x1_conv2d_linear_42 = model->params[35].values; 
inputs->weight_1x1_conv2d_43 = model->params[36].values; 
inputs->weight_depth_wise_43 = model->params[37].values; 
inputs->weight_1x1_conv2d_linear_43 = model->params[38].values; 
inputs->weight_1x1_conv2d_44 = model->params[39].values; 
inputs->weight_depth_wise_44 = model->params[40].values; 
inputs->weight_1x1_conv2d_linear_44 = model->params[41].values; 
inputs->weight_1x1_conv2d_45 = model->params[42].values; 
inputs->weight_depth_wise_45 = model->params[43].values; 
inputs->weight_1x1_conv2d_linear_45 = model->params[44].values; 
inputs->weight_1x1_conv2d_46 = model->params[45].values; 
inputs->weight_depth_wise_46 = model->params[46].values; 
inputs->weight_1x1_conv2d_linear_46 = model->params[47].values; 
inputs->weight_1x1_conv2d_47 = model->params[48].values; 
inputs->weight_depth_wise_47 = model->params[49].values; 
inputs->weight_1x1_conv2d_linear_47 = model->params[50].values; 
inputs->weight_1x1_conv2d_48 = model->params[51].values; 
inputs->weight_depth_wise_48 = model->params[52].values; 
inputs->weight_1x1_conv2d_linear_48 = model->params[53].values; 
inputs->data = model->input_vars[0].values; 
outputs->output = model->output_vars[0].values; 
}
#endif