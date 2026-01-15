import torch
import argparse
from exp.exp_main import Exp_Main

def arg_set(folder_path, data, model_name, pred_len=72, label_len=72, num_workers=10):
    """
    실험 인자 설정 함수
    
    Args:
        folder_path: 데이터 폴더 경로
        data: 데이터 파일 이름
        model_name: 모델 이름
        pred_len: 예측 길이 (기본값: 72)
        label_len: 레이블 길이 (기본값: 72)
        num_workers: 데이터 로더 워커 수 (기본값: 10)
    
    Returns:
        argparse.Namespace: 실험 설정 인자
    """
    args = argparse.Namespace(
        # 기본 설정
        is_training=1,
        train_only=False,
        model_id=f'{data}_{model_name}',
        model='Mixed_Linear',
        decomp_kernel_sizes=[25, 49],

        # 데이터 로더 설정
        data='custom',
        root_path=folder_path,
        data_path=data,
        features='M',
        target='현재수요(MW)',
        freq='5min',
        checkpoints='./checkpoints/',

        # 예측 작업 설정
        seq_len=96,
        label_len=label_len,
        pred_len=pred_len,

        individual=False,

        # 모델 설정
        embed_type=0,
        enc_in=8,
        dec_in=7,
        c_out=7,
        d_model=16,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=64,
        moving_avg=30,
        factor=1,
        distil=True,
        dropout=0.1,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        do_predict=True,

        # 최적화 설정
        num_workers=num_workers,
        itr=1,
        train_epochs=5,
        batch_size=16,
        patience=2,
        learning_rate=0.001,
        des='Exp',
        loss='mse',
        lradj='type1',
        use_amp=False,

        # GPU 설정
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices='0,1,2,3',
        test_flop=False
    )

    # channels 속성 추가
    args.channels = args.enc_in

    return args


def model_run(args):
    """
    모델 학습/평가/예측 실행 함수
    
    Args:
        args: argparse.Namespace, 실험 설정 인자
    """
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()


# 편의 함수들: 기존 호환성을 위한 래퍼 함수
def arg_set_12(folder_path, data, model_name):
    """pred_len=12, label_len=12 설정"""
    return arg_set(folder_path, data, model_name, pred_len=12, label_len=12)


def arg_set_36(folder_path, data, model_name):
    """pred_len=36, label_len=36 설정"""
    return arg_set(folder_path, data, model_name, pred_len=36, label_len=36)


def arg_set_72(folder_path, data, model_name):
    """pred_len=72, label_len=72 설정 (기본값)"""
    return arg_set(folder_path, data, model_name, pred_len=72, label_len=72)


def arg_set_144(folder_path, data, model_name):
    """pred_len=144, label_len=96 설정"""
    return arg_set(folder_path, data, model_name, pred_len=144, label_len=96, num_workers=4)
