import QtQuick
import QtQuick.Controls.Basic
ApplicationWindow {
    visible: true
    width: 800
    height: 700
    title: "HelloApp"
    Rectangle {
        anchors.fill: parent
        Image {
            sourceSize.width: parent.width
            sourceSize.height: parent.height
            source: "./images/Leafs.jpg"
            fillMode: Image.PreserveAspectCrop
        }
        Rectangle {
            anchors.fill: parent
            color: "transparent"
            Text {
                anchors {
                    bottom: parent.bottom
                    bottomMargin: 12
                    left: parent.left
                    leftMargin: 12
                }
                text: "Bring the Stadium to Your Home"
                font.pixelSize: 24
                color: "white"
            }
        }
    }
}