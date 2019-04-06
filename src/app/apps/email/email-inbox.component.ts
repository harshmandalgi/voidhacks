import { Component, OnInit, OnDestroy } from '@angular/core';
import { Email, EmailService } from './email.services';
import { ActivatedRoute, Router } from '@angular/router';
import { EmailComponent } from './email.component';


@Component ({
    templateUrl: 'email-inbox.html',
})

export class EmailInboxComponent implements OnInit, OnDestroy{
    emailItem: Email;

    private sub: any;
    
    constructor (
        private service : EmailService,
        private route: ActivatedRoute,
        private router: Router){          
    }
    
    ngOnInit() {
        this.sub = this.route.params.subscribe(params => {
            let id = params['id'];
            
            if (!id) {
                return;
            }
            
            //console.log('fetched email id: ' + id);
            this.service.getEmailItem(id).subscribe( emailItem => {
                this.emailItem = emailItem[id-1];
            });
        });
    }

    ngOnDestroy() {
        this.sub.unsubscribe();
    }
}