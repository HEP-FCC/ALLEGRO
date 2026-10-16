
import torch 

def load_train_model(args, dev):
    if args.load_model_weights is not None and args.correction:
        model = load_trainer(args, dev)
    elif args.load_model_weights is not None:
        ## from src.models.GATr.Gatr_withModifications import ExampleWrapper as GravnetModel
        from src.models.GATr.Gatr_pf_e_binary import ExampleWrapper as GravnetModel
        model = GravnetModel.load_from_checkpoint(
            args.load_model_weights, args=args, dev=0, map_location=dev,strict=False)
    return model 

def load_trainer(args, dev):
    ## from src.models.GATr.Gatr_withModifications import ExampleWrapper as GravnetModel
    from src.models.GATr.Gatr_pf_e_binary import ExampleWrapper as GravnetModel
    model = GravnetModel.load_from_checkpoint(
        args.load_model_weights, args=args, dev=0, map_location=dev,strict=False)
    print("weight before checkpoint", model.ec_model_wrapper_neutral.gatr.linear_out.s2mvs.weight)
    print("weight before checkpoint gatr", model.gatr.linear_out.s2mvs.weight)
    # TODO: evaluate the latest latest clustering!
    # model2 = GravnetModel.load_from_checkpoint("/mnt/proj2/dd-24-98/models/061024_cont2/_epoch=0_step=5500.ckpt", args=args, dev=0, strict=False)
    #model2 = GravnetModel.load_from_checkpoint(
    #        "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/181024_Hss/_epoch=4_step=50000.ckpt", args=args,
    #    dev=0, strict=False, map_location=torch.device("cuda:0"))  # Load the good clustering
    #model.gatr = model2.gatr
    #model.ScaledGooeyBatchNorm2_1 = model2.ScaledGooeyBatchNorm2_1
    #model.clustering = model2.clustering
    #model.beta = model2.beta
    #print("weight 2 after", model.ec_model_wrapper_neutral.gatr.linear_out.s2mvs.weight)
    #print("weight after checkpoint gatr", model.gatr.linear_out.s2mvs.weight)
    return model 


def load_test_model(args, dev):
    ## from src.models.GATr.Gatr_withModifications import ExampleWrapper as GravnetModel
    from src.models.GATr.Gatr_pf_e_binary import ExampleWrapper as GravnetModel
    model = GravnetModel.load_from_checkpoint(
        args.load_model_weights, args=args, dev=0, map_location=dev, strict=False
    )
    return model 

