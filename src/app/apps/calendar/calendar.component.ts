import { Component, AfterViewInit } from '@angular/core';

@Component ({
    templateUrl: 'calendar.html'
})

export class CalendarComponent {
    constructor() { }

    date: Date = new Date();
    d: any = new Date().getDate();
    m: any = new Date().getMonth();
    y: any = new Date().getFullYear();

    addEvent = function(calendarOptions) {
        calendarOptions.events.push({
            title: 'Untitle Event',
            start: new Date(this.y, this.m, 28),
            end: new Date(this.y, this.m, 29),
            desc: 'Meetings',
            className: ['openSesame']
        });
    };

    removeEvent = function(index, calendarOptions) {
        calendarOptions.events.splice(index,1);
    };

    calendarOptions = {
        height: 800,
        header:{
            left: 'month,agendaWeek,agendaDay',
            center: 'title',
            right: 'today prev,next'
        },
        fixedWeekCount : false,
        editable: true,
        eventLimit: true,
        events: [
            {
                title: 'All Day Event',
                start: new Date(this.y, this.m, 1),
                desc:'Meetings',
                bullet: 'success'
            },
            {
                title: 'Long Event',
                start: new Date(this.y, this.m, this.d - 5),
                end: new Date(this.y, this.m, this.d - 2),
                desc:'Hangouts',
                bullet: 'success'
            },
            {
                title: 'Repeating Event',
                start: new Date(this.y, this.m, this.d - 3, 16, 0),
                allDay: false,
                desc:'Product Checkup',
                bullet: 'warning'
            },
            {
                title: 'Repeating Event',
                start: new Date(this.y, this.m, this.d + 4, 16, 0),
                allDay: false,
                desc:'Conference',
                bullet: 'danger'
            },
            {
                title: 'Birthday Party',
                start: new Date(this.y, this.m, this.d + 1, 19, 0),
                end: new Date(this.y, this.m, this.d + 1, 22, 30),
                allDay: false,
                desc:'Gathering'
            },
            {
                title: 'Click for Google',
                url: 'http://google.com/',
                start: new Date(this.y, this.m, 28),
                end: new Date(this.y, this.m, 29),
                desc:'Google',
                bullet: 'success'
            }
        ]
    };
}