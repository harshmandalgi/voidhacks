import { Component } from '@angular/core';

@Component ({
    templateUrl: 'email-compose.html',
})

export class EmailComposeComponent {

    summernoteModel: string = 'content';

    multiSelectConfig : any = {
        create: true,
          valueField: 'id',
          labelField: 'title',
          delimiter: '|',
          placeholder: 'To:'
    };

    singleSelectValue: string[] = ['reactjs'];

    singleSelectOptions: any = [];
}