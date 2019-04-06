import {Component, OnInit } from '@angular/core';

@Component({
    templateUrl: 'progress-slider.html'
})
export class ProgressNSliderComponent  {
    constructor() { }

    public horizonSlider1: any = {
        start: 30,
        connect: 'lower',
        step: 1,
        range: {
            min: 0,
            max: 100
        }
    };
    public horizonSlider2: any = {
        start: 60,
        connect: 'lower',
        step: 1,
        range: {
            min: 0,
            max: 100
        }
    };

    public horizonSlider3: any = {
        start: 50,
        connect: 'lower',
        step: 1,
        range: {
            min: 0,
            max: 100
        }
    };

    public horizonSlider4: any = {
        start: 80,
        connect: 'lower',
        step: 1,
        range: {
            min: 0,
            max: 100
        }
    };

    public horizonSlider5: any = {
        start: 20,
        connect: 'lower',
        step: 1,
        range: {
            min: 0,
            max: 100
        }
    };

    public horizonSlider6: any = {
        start: 50,
        connect: 'lower',
        step: 1,
        range: {
            min: 0,
            max: 100
        }
    };

    public verticalSlider1: any = {
        start: 30,
        connect: 'lower',
        step: 1,
        orientation: 'vertical',
        range: {
            min: 0,
            max: 100
        }
    }; 

    public verticalSlider2: any = {
        start: 50,
        connect: 'lower',
        step: 1,
        orientation: 'vertical',
        range: {
            min: 0,
            max: 100
        }
    }; 

    public verticalSlider3: any = {
        start: 40,
        connect: 'lower',
        step: 1,
        orientation: 'vertical',
        range: {
            min: 0,
            max: 100
        }
    }; 

    public verticalSlider4: any = {
        start: 60,
        connect: 'lower',
        step: 1,
        orientation: 'vertical',
        range: {
            min: 0,
            max: 100
        }
    }; 

    public verticalSlider5: any = {
        start: 80,
        connect: 'lower',
        step: 1,
        orientation: 'vertical',
        range: {
            min: 0,
            max: 100
        }
    }; 

    public verticalSlider6: any = {
        start: 70,
        connect: 'lower',
        step: 1,
        orientation: 'vertical',
        range: {
            min: 0,
            max: 100
        }
    }; 
    
    public rangeSlider: any = {
        connect: true,
        step: 5,
        range: {
            min: 0,
            max: 1000
        }
    }; 

    public tabDragSlider: any = {
        connect: true,
        step: 5,
        behaviour: 'tap-drag',
        range: {
            min: 0,
            max: 1000
        }
    };

    public stepSlider: any = {
        connect: 'lower',
        step: 10,
        pips: { mode: 'steps', density: 10 },
        range: {
            min: 0,
            max: 100
        }
    };

    public sliderModel1: number = 30;
    public sliderModel2: number = 60;
    public sliderModel3: number = 50;
    public sliderModel4: number = 80;
    public sliderModel5: number = 20;
    public sliderModel6: number = 50;
    public verticalModel1: number = 30;
    public verticalModel2: number = 50;
    public verticalModel3: number = 40;
    public verticalModel4: number = 60;
    public verticalModel5: number = 80;
    public verticalModel6: number = 70;
    public rangeModel: number[] = [300, 800];
    public tabDragModel: number[] = [200, 500];
    public stepModel: number = 20;
}