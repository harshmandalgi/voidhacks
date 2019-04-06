import { NgModule } from '@angular/core';
import { Routes } from '@angular/router';

//Tables Components
import { FormElementsComponent } from './form-elements/form-elements.component';
import { FormLayoutsComponent } from './form-layouts/form-layouts.component';
import { FormWizard_Component } from './form-wizard/form-wizard.component';

export const FormsRoutes: Routes = [
    {
        path: '',
        children: [
            {
                path: 'form-elements',
                component: FormElementsComponent,
                data: {
                    title: 'Form Elements'
                }
            },
            {
                path: 'form-layouts',
                component: FormLayoutsComponent,
                data: {
                    title: 'Form Layouts'
                }
            },
            {
                path: 'form-wizard',
                component: FormWizard_Component,
                data: {
                    title: 'Form Wizard'
                }
            }
        ]
    }
];

