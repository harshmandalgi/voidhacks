import { Component } from '@angular/core';
import Chart from 'chart.js';
import { ThemeConstants } from '../../shared/config/theme-constant';

@Component ({
    templateUrl: 'chartjs.html'
})

export class ChartJsComponent {
    constructor(private colorConfig:ThemeConstants) { 
    }

    themeColors = this.colorConfig.get().colors;

    //Pie Chart Config
    public pieChartLabels:string[] = ['Download Sales', 'In-Store Sales', 'Mail Sales'];
    public pieChartData:number[] = [300, 300, 500];
    public pieChartType:string = 'pie';
    public pieChartLegend:boolean = false;
    public pieChartColors: any = [{ 
        backgroundColor: [this.themeColors.info, this.themeColors.primary, this.themeColors.success],
        pointBackgroundColor : [this.themeColors.info, this.themeColors.primary, this.themeColors.success]
    }];

    //Donut Chart Config
    public donutChartLabels:string[] = ['Download Sales', 'In-Store Sales', 'Mail Sales'];
    public donutChartData:number[] = [300, 500, 100];
    public donutChartType:string = 'doughnut';
    public donutChartLegend:boolean = false;
    public dountChartOptions:any = {
        cutoutPercentage: 75
    }
    public donutChartColors: any = [{ 
        backgroundColor: [this.themeColors.info, this.themeColors.dark, this.themeColors.success],
        pointBackgroundColor : [this.themeColors.info, this.themeColors.dark, this.themeColors.success]
    }];
    
    //Radar Chart Config
    public radarChartLabels:string[] = ["Eating", "Drinking", "Sleeping", "Designing", "Coding", "Cycling", "Running"];
    public radarChartData:any = [
        {data: [65, 59, 90, 81, 56, 55, 40], label: 'Series A'},
        {data: [28, 48, 40, 19, 96, 27, 100], label: 'Series B'}
    ];
    public radarChartLegend:boolean = false;
    public radarChartType:string = 'radar';
    public radarChartColors:Array<any> = [
        { 
            backgroundColor: this.themeColors.successInverse,
            borderColor: this.themeColors.success,
            pointBackgroundColor: this.themeColors.success
        },
        { 
            backgroundColor: this.themeColors.warningInverse,
            borderColor: this.themeColors.warning,
            pointBackgroundColor: this.themeColors.warning
        }
    ];

    //Polar Chart Config
    public polarChartLabels:string[] = ["Download Sales", "In-Store Sales", "Mail-Order Sales", "Tele Sales", "Corporate Sales"];
    public polarChartData:number[] = [300, 400, 100, 200, 100];        
    public polarChartType:string = 'polarArea';
    public polarChartLegend:boolean = false;
    public polarChartColors: any = [{ 
        backgroundColor: [this.themeColors.success, this.themeColors.info, this.themeColors.warning, this.themeColors.primary, this.themeColors.danger],
        borderColor : [this.themeColors.white, this.themeColors.white, this.themeColors.white, this.themeColors.white, this.themeColors.white]
    }];

    //Line Chart Config
    public lineChartLabels:Array<any> = ["16th", "17th", "18th", "19th", "20th", "21th", "22th"];
    public lineChartData:Array<any> = [
        {data: [65, 59, 80, 81, 56, 55, 40], label: 'Series A'},
        {data: [28, 48, 40, 19, 90, 27, 75], label: 'Series B'}
    ];
    public lineChartOptions:any = {
        scales: {
            yAxes: [
                {
                    id: 'y-axis-1',
                    type: 'linear',
                    display: true,
                    position: 'left'
                },
                {
                    id: 'y-axis-2',
                    type: 'linear',
                    display: false
                }
            ]
        },
        maintainAspectRatio: false
    };
    public lineChartLegend:boolean = false;
    public lineChartType:string = 'line';
    public lineChartColors:Array<any> = [
        {
            backgroundColor: this.themeColors.infoInverse,
            borderColor: this.themeColors.info
        },
        { 
            backgroundColor: this.themeColors.successInverse,
            borderColor: this.themeColors.success
        }
    ];

    //Bar Chart Config
    public barChartLabels:string[] = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'];
    public barChartType:string = 'bar';
    public barChartLegend:boolean = false;
    public barChartData:any[] = [
        {data: [65, 59, 80, 81, 56, 55, 40, 37, 54, 76, 63, 62], label: 'Series A'},
        {data: [28, 48, 40, 19, 86, 27, 90, 43, 65 ,76, 87, 85], label: 'Series B'}
    ];
    public barChartColors:Array<any> = [
        { 
            backgroundColor: this.themeColors.info,
            borderColor: this.themeColors.info
        },
        {
            backgroundColor: this.themeColors.primaryInverse,
            borderColor: this.themeColors.danger
        }
    ];
}