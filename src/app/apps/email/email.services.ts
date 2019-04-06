import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import 'rxjs/Rx'

export class Email {
    constructor(
        public id: number,
        public from : string,
        public subject: string,
        public body: string,
        public time: string,
        public to : string,
        public avatar: string,
        public files : any[],
        public type : string,
        public filename : string,
        public size : string
    ) { }
}


@Injectable()
export class EmailService {
    
    constructor(private http: HttpClient) {}

    getEmailData() {
        return this.http
        .get('../../../assets/js/data/email-list.json')
        .pipe(map(data => {data; return data;}))
    }

    getEmailItem(id: any[]) {
        let email = [];
        return this.http
        .get('../../../assets/js/data/email-list.json')
        .pipe(map(data => {data; return data;}))
        .pipe(map(this.flatEmailList))
        
    }
    
    flatEmailList(data){
        let flatData = [];
        data.emails.map(emails => flatData = flatData.concat(emails.list));
        return  flatData;
    }
}