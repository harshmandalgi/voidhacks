import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { ThemeConstants } from '../shared/config/theme-constant';
import { ChartsModule } from 'ng2-charts';
import 'd3';
import 'nvd3';
import { NvD3Module } from 'ng2-nvd3';

import { ChartsRoutes } from './charts.routing';

//Charts Component
import { ChartJsComponent } from './chartjs/chartjs.component';
import { SparkLineComponent } from './sparkline/sparkline.component';
import { Nvd3Component } from './nvd3/nvd3.component';


@NgModule({
    imports: [
        RouterModule.forChild(ChartsRoutes),
        ChartsModule,
        NvD3Module 
    ],
    declarations: [
        ChartJsComponent,
        SparkLineComponent,
        Nvd3Component
    ],
    providers: [
        ThemeConstants 
    ]
})
export class Charts_Module { }