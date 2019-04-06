import {Component} from '@angular/core';

@Component({
    templateUrl: 'form-wizard.html'
})
export class FormWizard_Component {
    constructor() { }

    public isCompleted: any;
    public onStep2Next: any;
    public onStep3Next: any;
    public onComplete: any;
}