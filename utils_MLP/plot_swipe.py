import torch
import plotsAnalysis
if __name__ == '__main__':
    pathnamelist = ['/scratch/yd105/ML_MM_Benchmark/MLP_CNN/models/color_dp_skip_1']
    for pathname in pathnamelist:
        
        # Forward: Convolutional swipe
        #plotsAnalysis.HeatMapBVL('kernel_first','kernel_second','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir='models/'+pathname,feature_1_name='kernel_first',feature_2_name='kernel_second')
        
        # General: Complexity swipe
        #plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + 'layer vs unit_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='linear_unit')
        # General: Complexity swipe
        #plotsAnalysis.HeatMapBVL('lr','reg_scale','lr vs reg_scale',save_name=pathname + 'lr vs reg_scale heatmap.png', HeatMap_dir=pathname,feature_1_name='lr',feature_2_name='reg_scale')
        # General: Complexity swipe
        #plotsAnalysis.HeatMapBVL('batch','lr_decay','batch vs lr_decay',save_name=pathname + 'batch vs lr_decay heatmap.png', HeatMap_dir=pathname,feature_1_name='batch_size',feature_2_name='lr_decay_rate')
        plotsAnalysis.HeatMapBVL('dropout','skip_head','dp vs skip',save_name=pathname + 'dp vs skip heatmap.png', HeatMap_dir=pathname,feature_1_name='dropout',feature_2_name='skip_head')
        

        # General: Complexity swipe
        #plotsAnalysis.HeatMapBVL('feature_ch_num','nhead_att','feature_ch_num vs nhead_att Heat Map',save_name=pathname + 'feature_ch_num vs nhead_att heatmap.png',
         #                       HeatMap_dir=pathname,feature_1_name='feature_channel_num',feature_2_name='nhead_encoder')
        
        # General: lr vs layernum
        #plotsAnalysis.HeatMapBVL('num_layers','lr','layer vs unit Heat Map',save_name=pathname + 'layer vs lr_heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='lr')

        # MDN: num layer and num_gaussian
        # plotsAnalysis.HeatMapBVL('num_layers','num_gaussian','layer vs num_gaussian Heat Map',save_name=pathname + 'layer vs num_gaussian heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='num_gaussian')
        
        # General: Reg scale and num_layers
        #plotsAnalysis.HeatMapBVL('num_layers','reg_scale','layer vs reg Heat Map',save_name=pathname + 'layer vs reg_heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='reg_scale')
        
        # # VAE: kl_coeff and num_layers
        # plotsAnalysis.HeatMapBVL('num_layers','kl_coeff','layer vs kl_coeff Heat Map',save_name=pathname + 'layer vs kl_coeff_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear_d',feature_2_name='kl_coeff')

        # # VAE: kl_coeff and dim_z
        # plotsAnalysis.HeatMapBVL('dim_z','kl_coeff','kl_coeff vs dim_z Heat Map',save_name=pathname + 'kl_coeff vs dim_z Heat Map heatmap.png',
        #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='kl_coeff')

        # # VAE: dim_z and num_layers
        # plotsAnalysis.HeatMapBVL('dim_z','num_layers','layer vs unit Heat Map',save_name=pathname + 'layer vs dim_z Heat Map heatmap.png',
        #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_d')
        
        # # VAE: dim_z and num_unit
        # plotsAnalysis.HeatMapBVL('dim_z','num_unit','dim_z vs unit Heat Map',save_name=pathname + 'dim_z vs unit Heat Map heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_unit')

        # # General: Reg scale and num_unit (in linear layer)
        # plotsAnalysis.HeatMapBVL('reg_scale','num_unit','reg_scale vs unit Heat Map',save_name=pathname + 'reg_scale vs unit_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='reg_scale',feature_2_name='linear_unit')
        
        # # cINN or INN: Couple layer num and lambda mse
        # plotsAnalysis.HeatMapBVL('couple_layer_num','lambda_mse','couple_num vs lambda mse Heat Map',save_name=pathname + 'couple_num vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='lambda_mse')
        
        # # cINN or INN: lambda z and lambda mse
        # plotsAnalysis.HeatMapBVL('lambda_z','lambda_mse','lambda_z vs lambda mse Heat Map',save_name=pathname + 'lambda_z vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='lambda_z',feature_2_name='lambda_mse')

        # # cINN or INN: lambda rev and lambda mse
        # plotsAnalysis.HeatMapBVL('lambda_rev','lambda_mse','lambda_rev vs lambda mse Heat Map',save_name=pathname + 'lambda_rev vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='lambda_rev',feature_2_name='lambda_mse')

        # # cINN or INN: lzeros_noise_scale and lambda mse
        # plotsAnalysis.HeatMapBVL('zeros_noise_scale','lambda_mse','zeros_noise_scale vs lambda mse Heat Map',save_name=pathname + 'zeros_noise_scale vs lambda mse_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='zeros_noise_scale',feature_2_name='lambda_mse')

        # # cINN or INN: zeros_noise_scaleand y_noise_scale
        # plotsAnalysis.HeatMapBVL('zeros_noise_scale','y_noise_scale','zeros_noise_scale vs y_noise_scale Heat Map',save_name=pathname + 'zeros_noise_scale vs y_noise_scale_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='zeros_noise_scale',feature_2_name='y_noise_scale')

        

        # # cINN or INN: Couple layer num and reg scale
        # plotsAnalysis.HeatMapBVL('couple_layer_num','reg_scale','layer vs unit Heat Map',save_name=pathname + 'couple_layer_num vs reg_scale_heatmap.png',
        #                          HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='reg_scale')
        
        
        # # INN: Couple layer num and dim_pad
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_tot','couple_layer_num vs dim pad Heat Map',save_name=pathname + 'couple_layer_num vs dim pad _heatmap.png',
        #                         HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='dim_tot')

        # # INN: Lambda_mse num and dim_pad
        # plotsAnalysis.HeatMapBVL('lambda_mse','dim_tot','lambda_mse vs dim_tot Heat Map',save_name=pathname + 'lambda_mse vs dim_tot_heatmap.png',
        #                         HeatMap_dir=pathname, feature_1_name='lambda_mse',feature_2_name='dim_tot')
        
        # # INN: Couple layer num and dim_z
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_z','couple_layer_num vs dim_z Heat Map',save_name=pathname + 'couple_layer_num vs dim_z_heatmap.png',
        #                        HeatMap_dir=pathname, feature_1_name='couple_layer_num',feature_2_name='dim_z')
