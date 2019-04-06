import { NgModule } from '@angular/core';
import { Routes } from '@angular/router';

//Charts Components
import { ChartJsComponent } from './chartjs/chartjs.component';
import { SparkLineComponent } from './sparkline/sparkline.component';
import { Nvd3Component } from './nvd3/nvd3.component';

export const ChartsRoutes: Routes = [
    {
        path: '',
        children: [
            {
                path: 'chartjs',
                component: ChartJsComponent,
                data: {
                    title: 'Chart Js'
                }
            },
            {
                path: 'sparkline',
                component: SparkLineComponent,
                data: {
                    title: 'Sparkline'
                }
            },
            {
                path: 'nvd3',
                component: Nvd3Component,
                data: {
                    title: 'nvd3'
                }
            }
        ]
    }
];

