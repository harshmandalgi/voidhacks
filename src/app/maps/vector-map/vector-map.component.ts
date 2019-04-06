// Typing for Ammap
/// <reference path="../../shared/typings/ammaps/ammaps.d.ts" />

import { Component, AfterViewInit } from '@angular/core';
import { ThemeConstants } from '../../shared/config/theme-constant';
import 'ammap3';
import 'ammap3/ammap/maps/js/worldLow';

@Component ({
    templateUrl: 'vector-map.html'
})

export class VectorMapComponent {

    constructor(private colorConfig:ThemeConstants) { 
    }

    ngAfterViewInit() {
        var map;
        let themeColors = this.colorConfig.get().colors;

        var targetSVG = "M9,0C4.029,0,0,4.029,0,9s4.029,9,9,9s9-4.029,9-9S13.971,0,9,0z M9,15.93 c-3.83,0-6.93-3.1-6.93-6.93S5.17,2.07,9,2.07s6.93,3.1,6.93,6.93S12.83,15.93,9,15.93 M12.5,9c0,1.933-1.567,3.5-3.5,3.5S5.5,10.933,5.5,9S7.067,5.5,9,5.5 S12.5,7.067,12.5,9z";
              
        if (AmCharts.isReady) {
            createChart();
        } 
        else {
            AmCharts.ready(function () {
                createChart();
            });
        }

        function createChart() {
            map = new AmCharts.AmMap();

            map.colorSteps = 0;

            map.imagesSettings = {
                rollOverColor: themeColors.info,
                rollOverScale: 1.5,
                selectedScale: 1.5,
                selectedColor: themeColors.info,
                color: themeColors.info
            }

            var dataProvider = {
                mapVar: AmCharts.maps.worldLow,

                images: [
                    {
                        svgPath: targetSVG,
                        zoomLevel: 5,
                        scale: 1,
                        title: "Malaysia (+25.17)",
                        latitude: 4.21,
                        longitude: 101.97
                    },
                    {
                        svgPath: targetSVG,
                        zoomLevel: 5,
                        scale: 1,
                        title: "China (-28.12)",
                        latitude: 37.09,
                        longitude: 95.71
                    },
                    {
                        svgPath: targetSVG,
                        zoomLevel: 5,
                        scale: 1,
                        title: "Japan (+18.84%)",
                        latitude: 36.20,
                        longitude: 138.25
                    },
                    {
                        svgPath: targetSVG,
                        zoomLevel: 5,
                        scale: 1,
                        title: "Brazil (+3.34%)",
                        latitude: -7.15,
                        longitude: -53.67
                    },
                    {
                        svgPath: targetSVG,
                        zoomLevel: 5,
                        scale: 1,
                        title: "United State (+16.68%)",
                        latitude: 40.02,
                        longitude: -104.01
                    },
                    {
                        svgPath: targetSVG,
                        zoomLevel: 5,
                        scale: 1,
                        title: "Greenland (+20.13%)",
                        latitude: 76.20,
                        longitude: -42.23
                    }
                ]
            };

            map.areasSettings = {
                autoZoom: true, 
                unlistedAreasColor: "#d7d8df",
                balloonText: "[[title]]<p>[[value]]</p> [[description]]",
                descriptionWindowHeight:300,
                descriptionWindowWidth:300,
                descriptionWindowTop:400,
                descriptionWindowLeft:300,
                outlineAlpha:1,
                outlineColor: themeColors.white,
                outlineThickness:1,
                rollOverOutlineColor: "#ffffff",
            };
            map.dataProvider = dataProvider;

            var valueLegend = new AmCharts.ValueLegend();
            valueLegend.right = 10;
            valueLegend.minValue = "";
            valueLegend.maxValue = "";
            map.valueLegend = valueLegend;

            map.write("mapdiv");
        }
    }
}