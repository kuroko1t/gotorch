package common

import "log"

func Log_Fatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
