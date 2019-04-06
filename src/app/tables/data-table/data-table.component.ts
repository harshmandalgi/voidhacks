import { Component, OnInit, ViewEncapsulation } from '@angular/core';

@Component ({
    templateUrl: 'data-table.html'
})

export class DataTableComponent implements OnInit {
    dtOptions: DataTables.Settings = {};
    
    ngOnInit(): void {
        this.dtOptions = {
            pagingType: 'full_numbers'
        };
    }
}