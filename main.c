#include <stdio.h>
#include <string.h>
#include "xtimer.h"
#include "random.h"
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <tvm/runtime/crt/logging.h>
#include "stdio_base.h"
#include "mlmci.h"

#ifndef UTOE_ONLY
#include "net/nanocoap_sock.h"
#endif

#ifndef UTOE_RANDOM_SEED
#define UTOE_RANDOM_SEED 42
#endif

#ifndef UTOE_TRIAL_NUM
#define UTOE_TRIAL_NUM 10
#endif

/* UTOE_GRANULARITY 0 - Per Model, 1 - Per Operator*/
#ifndef UTOE_GRANULARITY
#define UTOE_GRANULARITY 0
#endif

#define UTOE_RPC_BUFFER_SIZE 128

#ifndef UTOE_INPUT_SIZE
#define UTOE_INPUT_SIZE 4
#endif

#ifndef UTOE_OUTPUT_SIZE
#define UTOE_OUTPUT_SIZE 4
#endif

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
    (void) unused_context;
    stdio_write(data, size);
    return size;
}

#if (UTOE_GRANULARITY==0)
#include <tvmgen_default.h>
extern mlmodel_t *model_ptr;

void per_model_eval(void)
{       
    // #include "model_io_vars.h"
    (void) printf("U-TOE Per-Model Evaluation \n");
#ifdef UTOE_ONLY
    (void) printf("Press any key to start >\n");
    (void) getchar();
#endif
    SCB_EnableDCache();
    SCB_EnableICache();

    random_init(UTOE_RANDOM_SEED);
    uint32_t start, end;

    for(int i = 0; i < UTOE_TRIAL_NUM;i++) {

        for(int j = mlmodel_get_num_input_vars(model_ptr); j > 0; j--) {
            mlmodel_iovar_t *input = mlmodel_get_input_variable(model_ptr, j - 1);
            random_bytes(input->values, input->num_bytes);
        }
        
        start =  xtimer_now_usec();
        int ret_val = mlmodel_inference(model_ptr);
        end =  xtimer_now_usec();
        printf("trial: %d, usec: %ld, ret: %d \n", i, (long int)(end - start), ret_val);
    }
    (void) printf("Evaluation finished >\n");
}
#endif

#if (UTOE_GRANULARITY==1)
microtvm_rpc_server_t server;
void per_ops_eval(void)
{
    uint8_t buffer[UTOE_RPC_BUFFER_SIZE];
    random_init(UTOE_RANDOM_SEED);
    server = MicroTVMRpcServerInit(write_serial, NULL);
    TVMLogf("U-TOE Per-Operator Evaluation");
    TVMLogf("microTVM RIOT runtime - running");
    
    for(;;) {
        size_t bytes_read = stdio_read(buffer, UTOE_RPC_BUFFER_SIZE);
        uint8_t* arr_ptr = buffer;
        while (bytes_read > 0) {
            // Pass the received bytes to the RPC server.
            tvm_crt_error_t err = MicroTVMRpcServerLoop(server, &arr_ptr, &bytes_read);
            if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
                TVMPlatformAbort(err);
            }

        }
    }
}
#endif

#ifdef USE_SUIT
extern int suit_init(void);
#endif

#ifndef UTOE_ONLY
extern void coap_server_init();
#endif


int main(void)
{
    /* comment-out - causing no icmp echo */
    // xtimer_init();

#ifdef USE_SUIT
    suit_init();
#endif

#if (UTOE_GRANULARITY==1)
    per_ops_eval();
#elif (UTOE_GRANULARITY==0)
    mlmodel_init(model_ptr);
    mlmodel_set_global_model(model_ptr);

    per_model_eval();
#endif

#ifndef UTOE_ONLY

    printf("{\"IPv6 addresses\": [\"");
    netifs_print_ipv6("\", \"");
    puts("\"]}");
    
    coap_server_init();
#endif
    return 0;
}
