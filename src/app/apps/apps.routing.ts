import { NgModule } from '@angular/core';
import { Routes } from '@angular/router';

//Apps Components
import { SocialComponent } from './social/social.component';
import { CalendarComponent } from './calendar/calendar.component';
import { EmailComponent } from './email/email.component';
import { EmailInboxComponent } from './email/email-inbox.component';
import { EmailComposeComponent } from './email/email-compose.component';
import { TaskComponent } from './task/task.component'

export const AppsRoutes: Routes = [
    {
        path: '',
        children: [
            {
                path: 'social',
                component: SocialComponent,
                data: {
                    title: 'Social'
                }
            },
            {
                path: 'calendar',
                component: CalendarComponent,
                data: {
                    title: 'Calendar'
                }
            },
            {
                path: 'email',
                component: EmailComponent,
                data: {
                    title: 'Email'
                },
                children: [
                    {
                        path: 'email-inbox/:id',
                        component: EmailInboxComponent,
                        data: {
                            title: 'email-inbox'
                        }
                    },
                    {
                        path: 'email-compose',
                        component: EmailComposeComponent,
                        data: {
                            title: 'email-inbox'
                        }
                    }
                ]    
            },
            {
                path: 'task',
                component: TaskComponent,
                data: {
                    title: 'Task'
                }
            },
        ]
    }
];

