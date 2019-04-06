import { Component, AfterViewChecked } from '@angular/core';
import { ThemeConstants } from '../../shared/config/theme-constant';
import 'src/assets/js/jquery.sparkline/jquery.sparkline.js';
import * as $ from 'jquery';

@Component ({
    templateUrl: 'sparkline.html'
})

export class SparkLineComponent {

    constructor(private colorConfig:ThemeConstants) { 
    }
    
    ngAfterViewInit() {
        let themeColors = this.colorConfig.get().colors;

        // dummy data
        var sparklineBarData1 = [8, 10, 8, 10, 9, 10, 8, 9, 10, 10, 7, 10, 9, 7, 8, 8];
        var sparklineBarData2 = [0, 8, 10, 8, 10, 9, 10, 8,10, 10, 7, 10, 9,];
        var sparklineLineData1 = [32, 38, 36, 35, 38, 37, 35, 34, 36, 38, 36, 37, 35, 34, 37, 38, 38];
        var sparklineLineData2 = [7, 8, 10, 8, 10, 9, 10, 8,10, 10, 9, 10, 9,];
        var sparklinePieData1 = [4, 3, 2, 1];
        var sparklineTristateData1 = [1,1,0,1,-1,-1,1,-1,0,0,1,1 ];

        (<any>$('#bar-option-1') ).sparkline(sparklineBarData1,  
        {
            type: 'bar',
            height: '60',
            barWidth: 7,
            barSpacing: 5,
            barColor: themeColors.info
        });

        (<any>$("#bar-option-2")).sparkline( sparklineBarData2,  
        {
            type: 'bar',
            height: '100',
            barWidth: 7,
            barSpacing: 5,
            barColor: themeColors.success
        });

        (<any>$('#line-option-1') ).sparkline(sparklineLineData1,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '50',
            fillColor: themeColors.primaryInverse,
            lineColor: themeColors.primary,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });

        (<any>$('#line-option-2') ).sparkline(sparklineLineData2,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '50',
            fillColor: themeColors.successInverse,
            lineColor: themeColors.success,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });

        (<any>$('#line-option-3') ).sparkline(sparklineLineData1,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '50',
            fillColor: themeColors.primaryInverse,
            lineColor: themeColors.primary,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });

        (<any>$('#line-option-4') ).sparkline(sparklineLineData2,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '50',
            fillColor: themeColors.successInverse,
            lineColor: themeColors.success,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });

        (<any>$('#line-option-5') ).sparkline(sparklineLineData1,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '50',
            fillColor: themeColors.primaryInverse,
            lineColor: themeColors.primary,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });

        (<any>$('#line-option-6') ).sparkline(sparklineLineData2,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '50',
            fillColor: themeColors.successInverse,
            lineColor: themeColors.success,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });

        (<any>$("#pie-option")).sparkline( sparklinePieData1,  
        {
            type: 'pie',
            height: '100',
            sliceColors: [themeColors.danger, themeColors.success, themeColors.info, themeColors.warning]
        });
    
        (<any>$("#line-option-7")).sparkline( sparklineLineData2,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '100',
            fillColor: 'rgba(0,0,0,0)',
            lineColor: themeColors.warning,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });
    
        (<any>$("#discrete-option")).sparkline( sparklineLineData1,  
        {
            type: 'discrete',
            width: '130',
            lineColor: themeColors.primary,
            height: '100',
            lineHeight: 60
        });

        (<any> $("#tristate-option")).sparkline( sparklineTristateData1,  
        {
            type: 'tristate',
            width: '130',
            height: '100', 
            posBarColor: themeColors.success,
            negBarColor: themeColors.danger,
            barWidth: 7,
            barSpacing: 1
        });
    
        (<any>$("#line-option-8")).sparkline( sparklineLineData1,  
        {
            type: 'line',
            width: '130',
            spotColor: false,
            minSpotColor: false,
            maxSpotColor:false,
            lineWidth: 1,
            height: '100',
            fillColor: themeColors.dangerInverse,
            lineColor: themeColors.danger,
            highlightLineColor: 'rgba(0,0,0,.09)',
            highlightSpotColor: 'rgba(0,0,0,.21)',
        });
    }    
}