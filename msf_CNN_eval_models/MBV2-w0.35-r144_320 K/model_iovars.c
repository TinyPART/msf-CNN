/* U-TOE Generated File */ 
#include <stddef.h>
#include <stdint.h>
#include "mlmci.h"
__attribute__ ((aligned(4)))
static uint8_t data[62208]; 
__attribute__ ((aligned(4)))
static uint8_t output[11200]; 
static const mlmodel_iovar_t _model_input_vars[] = { 
{.name = "data",.values = (uint8_t*)data,.num_bytes = sizeof(data),},
};
static const mlmodel_iovar_t _model_output_vars[] = { 
{.name = "output",.values = (uint8_t*)output,.num_bytes = sizeof(output),},
};
const mlmodel_iovar_t* get_model_input_vars(void) {
return _model_input_vars;
}
const mlmodel_iovar_t* get_model_output_vars(void) {
return _model_output_vars;
}
