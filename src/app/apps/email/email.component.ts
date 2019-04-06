import { Component, OnInit, OnDestroy } from '@angular/core';
import { Email, EmailService } from './email.services';
import { Observable } from 'rxjs/Observable';
import { EmailInboxComponent } from './email-inbox.component';
import { Router, ActivatedRoute, ParamMap } from '@angular/router';


@Component ({
    templateUrl: 'email.html',
})

export class EmailComponent implements OnInit {

    public emailLayout : any;
    public emails : any[];
    private sub: any;
    private selectedId: number;

    constructor (
        private service : EmailService,
        private route: ActivatedRoute,
        private router: Router){  
    }

    ngOnInit() {
        this.service.getEmailData().subscribe((emails: any) => {
            this.emails = emails.emails; 
            //console.log(this.emails)
       }); 

       this.emailLayout = {
            emailViewMobile: false,
            emailNavMobile: false
        }; 
    }

    formatBody = function(body) {
        return body.replace(/<(?:.|\n)*?>/gm, '');
    }

    isSelected(emailItem: Email) { 
        return emailItem.id == this.selectedId;       
    }
    
    onSelect(emailItem: Email) {
        this.router.navigate(['/apps/email/email-inbox', emailItem.id]);
    }
}