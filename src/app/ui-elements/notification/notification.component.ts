import { Component, AfterViewInit, ViewEncapsulation} from '@angular/core';
import { Subject, Observable, Subscription } from 'rxjs/Rx';
import { ToastyService, ToastyConfig, ToastOptions, ToastData } from 'ng2-toasty';
import { NotificationCommunicationService } from './notification.services';
import swal from 'sweetalert2'


@Component({
    templateUrl: 'notification.html'
})
export class NotificationComponent implements AfterViewInit {
    public toastyComponentPosition: any;

    themes = [
        { name: 'Default Theme', code: 'default' }, 
        { name: 'Material Design', code: 'material' }, 
        { name: 'Bootstrap 3',code: 'bootstrap' }
    ];

    types = [
        { name: 'Default', code: 'default' }, 
        { name: 'Info', code: 'info' }, 
        { name: 'Success', code: 'success' }, 
        { name: 'Wait', code: 'wait' }, 
        { name: 'Error', code: 'error' }, 
        { name: 'Warning', code: 'warning'}
    ];

    positions = [
        { name: 'Top Left', code: 'top-left' }, 
        { name: 'Top Center', code: 'top-center' }, 
        { name: 'Top Right', code: 'top-right' }, 
        { name: 'Bottom Left', code: 'bottom-left' }, 
        { name: 'Bottom Center', code: 'bottom-center' }, 
        { name: 'Bottom Right', code: 'bottom-right' }, 
        { name: 'Center Center', code: 'center-center'}
    ];

    position: string = this.positions[5].code;

    options = {
        title: 'Toast It!',
        msg: 'Mmmm, tasties...',
        showClose: true,
        timeout: 5000,
        theme: this.themes[0].code,
        type: this.types[0].code
    };

    getTitle(num: number): string {
        return 'Countdown: ' + num;
    }

    getMessage(num: number): string {
        return 'Seconds left: ' + num;
    }

    constructor(private toastyService: ToastyService, private toastCommunicationService: NotificationCommunicationService) { 
      
    }
    
    ngAfterViewInit() {
        // this._toastyConfig.theme = 'bootstrap';
        let toastOptions: ToastOptions = {
            title: 'title',
            msg: 'my message',
            showClose: true,
            timeout: 15000,
            theme: "bootstrap",
            onAdd: (toast: ToastData) => {
                console.log('Toast ' + toast.id + ' has been added!');
            },
            onRemove: function (toast: ToastData) {
                console.log('Toast ' + toast.id + ' has been removed!');
            }
        };
        this.toastyService.success(toastOptions);
    }

    newToast() {
        let toastOptions: ToastOptions = {
            title: this.options.title,
            msg: this.options.msg,
            showClose: this.options.showClose,
            timeout: this.options.timeout,
            theme: this.options.theme,
            onAdd: (toast: ToastData) => {
                console.log('Toast ' + toast.id + ' has been added!');
            },
            onRemove: function(toast: ToastData) {
                console.log('Toast ' + toast.id + ' has been removed!');
            }
        };

        switch (this.options.type) {
            case 'default': this.toastyService.default(toastOptions); break;
            case 'info': this.toastyService.info(toastOptions); break;
            case 'success': this.toastyService.success(toastOptions); break;
            case 'wait': this.toastyService.wait(toastOptions); break;
            case 'error': this.toastyService.error(toastOptions); break;
            case 'warning': this.toastyService.warning(toastOptions); break;
        }
    }

    newCountdownToast() {
        let interval = 1000;
        let seconds = this.options.timeout / 1000;
        let subscription: Subscription;

        let toastOptions: ToastOptions = {
            title: this.getTitle(seconds || 0),
            msg: this.getMessage(seconds || 0),
            showClose: this.options.showClose,
            timeout: this.options.timeout,
            theme: this.options.theme,
            onAdd: (toast: ToastData) => {
                console.log('Toast ' + toast.id + ' has been added!');
                // Run the timer with 1 second iterval
                let observable = Observable.interval(interval).take(seconds);
                // Start listen seconds bit
                subscription = observable.subscribe((count: number) => {
                    // Update title
                    toast.title = this.getTitle(seconds - count - 1 || 0);
                    // Update message
                    toast.msg = this.getMessage(seconds - count - 1 || 0);
                });

            },
            onRemove: function(toast: ToastData) {
                console.log('Toast ' + toast.id + ' has been removed!');
                // Stop listenning
                subscription.unsubscribe();
            }
        };

        switch (this.options.type) {
            case 'default': this.toastyService.default(toastOptions); break;
            case 'info': this.toastyService.info(toastOptions); break;
            case 'success': this.toastyService.success(toastOptions); break;
            case 'wait': this.toastyService.wait(toastOptions); break;
            case 'error': this.toastyService.error(toastOptions); break;
            case 'warning': this.toastyService.warning(toastOptions); break;
        }
    }

    clearToasties() {
        this.toastyService.clearAll();
    }

    changePosition($event) {
        this.position = $event;
        // Update position of the Toasty Component
        this.toastCommunicationService.setPosition(this.position);
    }

    sweetAlert1() {
        swal(
            "Here's a message"
        )
    }

    sweetAlert2() {
        swal(
            "Here's a message!", "It's pretty, isn't it?"
        )
    }

    sweetAlert3() {
        swal(
            "Good job!", "You clicked the button!", "success"
        )
    }

    sweetAlert4() {
        swal({
            title: "Are you sure?",   
            text: "Your will not be able to recover this imaginary file!",   
            type: "warning",   
            showCancelButton: true,   
            confirmButtonColor: "#DD6B55",   
            confirmButtonText: "Yes, delete it!"
        }).then(function() {
            swal(
                "Booyah!"
            );
        })
    }

    sweetAlert5() {
        swal({
            title: 'Are you sure?',
            text: "You won't be able to revert this!",
            type: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#3085d6',
            cancelButtonColor: '#d33',
            confirmButtonText: 'Yes, delete it!',
            cancelButtonText: 'No, cancel!',
            confirmButtonClass: 'btn btn-success',
            cancelButtonClass: 'btn btn-danger',
            buttonsStyling: false
        }).then(function () {
            swal(
                'Deleted!',
                'Your file has been deleted.',
                'success'
            )
        }, function (dismiss) {
            // dismiss can be 'cancel', 'overlay',
            // 'close', and 'timer'
            if (dismiss === 'cancel') {
                swal(
                    'Cancelled',
                    'Your imaginary file is safe :)',
                    'error'
                )
            }
        })
    }
}