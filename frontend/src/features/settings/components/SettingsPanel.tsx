import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../../components/ui/card';
import { Input } from '../../../components/ui/input';
import { Label } from '../../../components/ui/label';
import { Switch } from '../../../components/ui/switch';
import { Badge } from '../../../components/ui/badge';
import { Select, NativeSelect, SelectItem } from '../../../components/ui/select';
import { useLanguage } from '../../../i18n/LanguageContext';

export function SettingsPanel() {
  const { t, locale } = useLanguage();

  const monitoredClasses = locale === 'es'
    ? ['Empujón', 'Golpe', 'Caída', 'Cuchillo', 'Pistola']
    : ['Push', 'Hit', 'Fall', 'Knife', 'Gun'];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>{t.settings.generalTitle}</CardTitle>
          <CardDescription>{t.settings.generalDesc}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">{t.settings.pushNotifications}</div>
              <div className="text-xs text-slate-500">{t.settings.pushNotificationsDesc}</div>
            </div>
            <Switch defaultChecked />
          </div>
          <div>
            <Label>{t.settings.timezone}</Label>
            <Select defaultValue="America/Mexico_City">
              <NativeSelect>
                <SelectItem value="America/Mexico_City">America/Mexico_City</SelectItem>
                <SelectItem value="UTC">UTC</SelectItem>
              </NativeSelect>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t.settings.faceTitle}</CardTitle>
          <CardDescription>{t.settings.faceDesc}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>{t.settings.similarityThreshold}</Label>
            <Input type="number" defaultValue={0.52} step="0.01" />
            <p className="text-xs text-slate-500">{t.settings.similarityHint}</p>
          </div>
          <div className="space-y-1">
            <Label>{t.settings.normalization}</Label>
            <Select defaultValue="ArcFace">
              <NativeSelect>
                <SelectItem value="ArcFace">Hiperesfera (ArcFace)</SelectItem>
                <SelectItem value="CosFace">CosFace</SelectItem>
              </NativeSelect>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t.settings.poseTitle}</CardTitle>
          <CardDescription>{t.settings.poseDesc}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>{t.settings.temporalWindow}</Label>
            <Input type="number" defaultValue={64} />
          </div>
          <div className="space-y-1">
            <Label>{t.settings.monitoredClasses}</Label>
            <div className="flex flex-wrap gap-1">
              {monitoredClasses.map((c) => (
                <Badge key={c} variant="outline" className="rounded-full">{c}</Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}