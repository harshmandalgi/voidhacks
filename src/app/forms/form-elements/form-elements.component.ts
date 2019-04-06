import { Component, OnInit } from '@angular/core';
import { NgbDateStruct } from '@ng-bootstrap/ng-bootstrap';
import { ViewEncapsulation } from '@angular/core';

const now = new Date();

@Component ({
    templateUrl: 'form-elements.html',
    styleUrls: ['../../../assets/scss/plugins/_datepicker.scss']
})

export class FormElementsComponent implements OnInit {
    constructor() { }

    public navigation: any;

    singleSelectOptions: any = [
        {
            label: 'Angular',
            value: 'angular',
            code: 'NG'
        }, {
            label: 'ReactJS',
            value: 'reactjs',
            code: 'RJS'
        }, {
            label: 'Ember JS',
            value: 'emberjs',
            code: 'emjs'
        }, {
            label: 'Ruby on Rails',
            value: 'ruby_on_rails',
            code: 'ROR'
        }
    ];

    singleSelectConfig: any = {
        labelField: 'label',
        valueField: 'value',
        searchField: ['label']
    };

    singleSelectValue: string[] = ['reactjs'];

    multiSelectOptions: any = [
        {
            label: 'Angular',
            value: 'angular',
            code: 'NG'
        }, {
            label: 'ReactJS',
            value: 'reactjs',
            code: 'RJS'
        }, {
            label: 'Ember JS',
            value: 'emberjs',
            code: 'emjs'
        }, {
            label: 'Ruby on Rails',
            value: 'ruby_on_rails',
            code: 'ROR'
        }
    ];

    multiSelectConfig: any = {
        labelField: 'label',
        valueField: 'value',
        searchField: ['label'],
        maxItems: 4
    };

    multiSelectValue: string[] = ['reactjs'];
    multiSelectValue2: string[] = ['reactjs', 'angular'];


    optGroupConfig: any = {
        maxItems: 1,
        optgroupField: 'class',
          labelField: 'name',
          searchField: ['name'],
          render: {
            optgroup_header: function(data, escape) {
                return '<div class="optgroup-header">' +
                        escape(data.label) +
                        ' <span class="scientific">' +
                        escape(data.label_scientific) +
                        '</span></div>';
            }
        },
        optgroups: [
            {value: 'mammal', label: 'Mammal', label_scientific: 'Mammalia'},
            {value: 'bird', label: 'Bird', label_scientific: 'Aves'},
            {value: 'reptile', label: 'Reptile', label_scientific: 'Reptilia'}
        ]
    };

    optGroupOptions: any = [
        {class: 'mammal', value: 'dog', name: 'Dog' },
        {class: 'mammal', value: 'cat', name: 'Cat' },
        {class: 'mammal', value: 'horse', name: 'Horse' },
        {class: 'mammal', value: 'kangaroo', name: 'Kangaroo' },
        {class: 'bird', value: 'duck', name: 'Duck'},
        {class: 'bird', value: 'chicken', name: 'Chicken'},
        {class: 'bird', value: 'ostrich', name: 'Ostrich'},
        {class: 'bird', value: 'seagull', name: 'Seagull'},
        {class: 'reptile', value: 'snake', name: 'Snake'},
        {class: 'reptile', value: 'lizard', name: 'Lizard'},
        {class: 'reptile', value: 'alligator', name: 'Alligator'},
        {class: 'reptile', value: 'turtle', name: 'Turtle'}
    ];

    optGroupValue: string[] = ['dog'];


    model;
    model2;
    model3;

    date: {year: number, month: number};
    displayMonths = 2;

    selectToday() {
        this.model = {year: now.getFullYear(), month: now.getMonth() + 1, day: now.getDate()};
    }


    ngOnInit() {
    }
}
