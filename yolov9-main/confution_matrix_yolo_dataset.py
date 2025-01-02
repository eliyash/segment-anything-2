from val_dual import parse_opt, ROOT, main


if __name__ == "__main__":
    opt = parse_opt()
    opt.data = ROOT / 'data/chimp_id.yaml'
    opt.weights = r'C:\Users\Eliahu\Downloads\best_finetun_chimp_id_on_ccr.pt'
    opt.project = r'D:\inference\confusion_best_finetun_chimp_id_on_ccr'
    main(opt)
