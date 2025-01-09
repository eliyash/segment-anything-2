from val_dual import parse_opt, ROOT, main


if __name__ == "__main__":
    opt = parse_opt()
    opt.data = ROOT / 'data/chimp.yaml'
    # opt.weights = r'C:\Users\Eliahu\Downloads\best_finetun_chimp_id_on_ccr.pt'
    opt.weights = r'C:\Users\Eliahu\Downloads\best_trained_on_ccr.pt'
    opt.project = r'D:\inference\count_crop_and_recognise_dataset'
    main(opt)
