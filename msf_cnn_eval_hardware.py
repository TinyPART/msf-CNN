from connector import get_local_controller, get_fit_iotlab_controller
import json
import os
import copy
import glob
import shutil
from evaluate import parse_per_model_output

def main():
    MODELS_PATH = "./msf_CNN_eval_models/"
    board = "nucleo-f767zi"
    os.environ['UTOE_ONLY'] = '1'
    env = {'BOARD': board, 'UTOE_TRIAL_NUM': str(1), 'UTOE_RANDOM_SEED': str(42),
        #    'PORT': "/dev/ttyACM0",
           }    

    hil_rslts = []

    with open(MODELS_PATH + "result.json", "r") as f:
        anly_rslts = json.load(f)

    for rslt in anly_rslts:
        model_dir_name = f'{rslt["Model"]}_{rslt["Configuration"]}'
        model_dir_path = os.path.join(MODELS_PATH, model_dir_name)
        for file in glob.glob(model_dir_path + r'/model_*'):
            print(file)
            shutil.copy(file, "./")
        shutil.copy(model_dir_path + "/default.tar", "./models/default/default.tar")
        riot_ctrl = get_local_controller(env)
        process_obj = riot_ctrl.flash(stdout=None, stderr=None)
        if process_obj.returncode == 0:
            term_retry_times = 2
            with riot_ctrl.run_term(reset=True): #reset should be false for risc v
                while term_retry_times > 0 :
                    try:
                        # riot_ctrl.term.expect_exact('start >')
                        riot_ctrl.term.sendline('s')
                        riot_ctrl.term.expect_exact('finished >',timeout=30)
                        break
                    except:
                        print("Exception Occured, term buffer:")
                        print(riot_ctrl.term.before)
                        term_retry_times -= 1
                        print("Retrying...")
            raw_output = riot_ctrl.term.before
            riot_ctrl.stop_exp()
        else:
            raw_output = ""
        eval_rslt = parse_per_model_output(raw_output)
        new_rslt = copy.deepcopy(rslt)
        new_rslt['eval_record'] = eval_rslt
        if len(eval_rslt['usec']) > 0:
            new_rslt['usec'] = eval_rslt['usec'][-1]
        else:
            new_rslt['usec'] = -1
        hil_rslts.append(new_rslt)
        with open(f"./msf_cnn_HIL_eval_result_{board}.json", "w") as f:
            json.dump(hil_rslts, f)
    

if __name__ == "__main__":
    main()