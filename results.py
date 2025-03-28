import json, os
def compute_results(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    n_ims = len(data)
    return n_ims

def check_results():
    im_names_result = []
    im_names = os.listdir("./ims_68_ver_0/test")
    print("im_names", im_names)
    json_file = "./ims_68_ver_0/result/deepface_ArcFace_correct.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    im_names_result += [x["image"] for x in data]

    json_file = "./ims_68_ver_0/result/deepface_ArcFace_incorrect.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    im_names_result += [x["image"] for x in data]

    print("im_names_result", im_names_result, len(im_names_result))
    for im_name_result in im_names_result:
        if im_name_result not in im_names:
            print(im_name_result)

if __name__ == "__main__":
    results_dir = [f"ims_68_ver_{i}/result" for i in range(5)]
    for result_dir in results_dir:
        print('result_dir', result_dir)
        for json_name in os.listdir(result_dir):
            json_file = os.path.join(result_dir, json_name)
            if not json_file.endswith(".json"):
                continue
            n_ims = compute_results(json_file)
            print(f"Json files: {result_dir}_{json_name}, n_ims, {n_ims}")
    
    print("Done!", len(os.listdir("./ims_68_ver_4/test")))
    # check_results()